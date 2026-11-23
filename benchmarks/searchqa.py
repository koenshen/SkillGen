"""Run native SkillGen with the fixed SkillOpt SearchQA train, dev and test sets."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import string

from models import SkillStatus, TaskInstance, TaskType

# SHA-256 of newline-joined item IDs, in the order used in the fixed run.
SPLIT_SPEC = {
    "train": (400, "cded018312f19f27e1da3e019931fb4d18fbbdbc4fd6d1651b5a52f29dedd268"),
    "val": (200, "d2913aff1ba10bd9a577e79c24121108218b30136c0d5a7002eed36133babafe"),
    "test": (1400, "7fca98256e96c837eee2612da9b9d29fb9d4e5cb908c751ca9bfdfd2d10ab6fb"),
}


def load_splits(split_dir=None):
    default_data = Path(__file__).resolve().parents[2] / "SkillOpt/data/searchqa_split"
    data = Path(split_dir or os.environ.get("SEARCHQA_SPLIT_DIR") or default_data)
    splits = {}
    for split, (count, fingerprint) in SPLIT_SPEC.items():
        path = data / split / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids = [row["id"] for row in rows]
        if len(ids) != count or hashlib.sha256("\n".join(ids).encode()).hexdigest() != fingerprint:
            raise ValueError(f"{path} must contain the {count} fixed SearchQA IDs in release order")
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "context")):
                raise ValueError(f"Missing question or context in {split}/{row['id']}")
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"Invalid reference answers in {split}/{row['id']}")
        splits[split] = rows
    return splits


def format_question(row):
    context = row["context"]
    if len(context) > 6000:
        kept = ""
        for passage in context.split("[DOC]"):
            with_passage = kept + "[DOC]" + passage if kept else passage
            if len(with_passage) > 6000:
                break
            kept = with_passage
        context = kept or (context[:6000] + "\n...[truncated]")
    return f"## Context\n{context}\n\n## Question\n{row['question']}"


def _answer_tokens(answer):
    text = "".join(c for c in answer.lower() if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def score_response(response, answers):
    tags = re.findall(r"<answer>(.*?)</answer>", response, re.I | re.S)
    if tags:
        answer = tags[-1].strip()
    else:
        lines = response.strip().splitlines()
        answer = lines[-1].strip() if lines else ""
    pred = _answer_tokens(answer)
    em = best_f1 = 0.0
    for reference in answers:
        gold = _answer_tokens(reference)
        if pred == gold:
            em = best_f1 = 1.0
            break
        common = sum((Counter(pred) & Counter(gold)).values())
        if common:
            precision, recall = common / len(pred), common / len(gold)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return {"em": em, "f1": best_f1, "predicted_answer": answer}


def instances(rows):
    return [
        TaskInstance(
            instance_id=row["id"],
            input=format_question(row),
            ground_truth=row["answers"],
            metadata={"benchmark": "searchqa"},
        )
        for row in rows
    ]


def test_metrics(result, expected_ids):
    trajectories = result.trajectories
    if len(trajectories) != len(expected_ids) or {t.instance_id for t in trajectories} != expected_ids:
        raise ValueError("SearchQA test trajectories do not match the fixed test IDs")
    em = f1 = 0.0
    errors = 0
    for traj in trajectories:
        if traj.metadata.get("agent_exception"):
            errors += 1
            continue
        grade = traj.metadata["searchqa_eval"]
        em += grade["em"]
        f1 += grade["f1"]
    n = len(expected_ids)
    return {"n": n, "em": em / n, "f1": f1 / n, "agent_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", help="Materialized SkillOpt SearchQA split directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skill-repo", help="Test an existing SkillGen output directory (skip training)")
    parser.add_argument("--output", type=Path, help="New directory for held-out test results")
    parser.add_argument("--check", action="store_true", help="Validate fixed data only, with no model calls")
    args = parser.parse_args()
    splits = load_splits(args.split_dir)
    if args.check:
        print("SearchQA:", {split: len(rows) for split, rows in splits.items()})
        return
    if args.output is None:
        parser.error("--output is required for training or testing")
    args.output.mkdir(parents=True, exist_ok=False)

    import llm
    from artifacts import write_json, write_trajectories
    from eval_skill import load_skill, run_condition
    from logging_utils import configure_logging
    from pipeline import load_config, run_pipeline

    configure_logging()
    cfg = load_config(args.config)
    if args.skill_repo:
        skill = load_skill(args.skill_repo)
    else:
        skill = run_pipeline(
            instances(splits["train"]), TaskType.BINARY,
            verification_instances=instances(splits["val"]),
            config_path=args.config, dataset_id="searchqa", task_name="SearchQA",
        )

    model_cfg = cfg["models"]
    model = model_cfg.get("verification_agent", model_cfg["default"])
    judge = model_cfg.get("verification_judge", model_cfg["default"])
    test = instances(splits["test"])
    expected_ids = {inst.instance_id for inst in test}
    workers = cfg["pipeline"].get("max_workers", 16)
    llm.reset_token_stats()
    with llm.stage_scope("test_baseline"):
        baseline = run_condition(test, TaskType.BINARY, model, judge, None, workers)
    write_trajectories(args.output / "baseline.jsonl", baseline.trajectories)
    baseline_metrics = test_metrics(baseline, expected_ids)
    deployed = bool(skill and skill.status != SkillStatus.DEPRECATED and (skill.body or "").strip())
    final_metrics = baseline_metrics
    if deployed:
        with llm.stage_scope("test_skill"):
            final = run_condition(test, TaskType.BINARY, model, judge, skill, workers,
                                  execute_scripts=bool(skill.scripts))
        write_trajectories(args.output / "skill.jsonl", final.trajectories)
        final_metrics = test_metrics(final, expected_ids)
    llm.dump_token_stats(args.output / "test_token_usage.json")
    summary = {"model": model, "skill_id": skill.skill_id if skill else None,
               "skill_deployed": deployed, "baseline": baseline_metrics, "final": final_metrics}
    write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
