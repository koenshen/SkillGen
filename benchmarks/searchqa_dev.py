"""Batch only the candidate validation calls for fixed SearchQA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from models import TaskInstance, Trajectory
from trajectory import check_searchqa_rollouts


@dataclass
class DevEarlyStop:
    order: str = "hard-first"
    batch_size: int = 4
    early_accept: bool = False
    # Insertion order is the canonical dev manifest, including for tied scores.
    history: dict[str, list[bool]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.order not in ("hard-first", "easy-first") or self.batch_size < 1:
            raise ValueError("Use hard-first or easy-first and a positive dev batch size")

    def seed(self, instances: list[TaskInstance], baseline: dict[str, Trajectory]):
        ids = [item.instance_id for item in instances]
        if not ids or len(set(ids)) != len(ids) or set(ids) != set(baseline):
            raise ValueError("Expected one scored baseline trajectory for every fixed dev ID")
        self.history = {i: [bool(baseline[i].success)] for i in ids}

    def evaluate(self, instances: list[TaskInstance], required_gain: int,
                 run_batch: Callable[[list[TaskInstance]], list[Trajectory]]):
        by_id = {item.instance_id: item for item in instances}
        if not self.history or set(self.history) != set(by_id) or len(by_id) != len(instances):
            raise ValueError("Candidate IDs differ from the fixed dev baseline")
        n = len(by_id)
        baseline_correct = sum(obs[0] for obs in self.history.values())
        threshold = baseline_correct + required_gain
        ordered = sorted(self.history, key=lambda i: sum(self.history[i]) / len(self.history[i]),
                         reverse=self.order == "easy-first")
        measured: list[Trajectory] = []
        batches = []
        correct = 0
        reject_certified_at = None
        for start in range(0, n, self.batch_size):
            ids = ordered[start:start + self.batch_size]
            batch = run_batch([by_id[i] for i in ids])
            if len(batch) != len(ids) or {t.instance_id for t in batch} != set(ids):
                raise ValueError("Incomplete or duplicate candidate dev results in a batch")
            measured.extend(batch)
            correct += sum(bool(t.success) for t in batch)
            upper = correct + n - len(measured)
            batches.append({"n_evaluated": len(measured), "correct_lower": correct,
                            "correct_upper": upper})
            if reject_certified_at is None and upper < threshold:
                reject_certified_at = len(measured)
            # A certified failure still needs all case feedback for refinement.
            if self.early_accept and correct >= threshold:
                break
        # Apply the native backend-failure guard to the whole observed candidate,
        # not to one four-item batch. Router bypasses made no candidate request.
        check_searchqa_rollouts([t for t in measured if not t.agent_config.get("skill_bypassed_by_router")])
        for traj in measured:
            self.history[traj.instance_id].append(bool(traj.success))
        stopped = len(measured) < n
        report = {
            "order": self.order, "batch_size": self.batch_size, "early_accept": self.early_accept,
            "n_total": n, "n_evaluated": len(measured), "stopped": stopped,
            "reason": "certified_pass" if stopped else "full_evaluation",
            "baseline_correct": baseline_correct, "required_correct": threshold,
            "reject_certified_at": reject_certified_at,
            "correct_lower": correct, "correct_upper": upper,
            "hard_lower": correct / n, "hard_upper": upper / n,
            "net_gain_lower": correct - baseline_correct, "net_gain_upper": upper - baseline_correct,
            "evaluated_ids": ordered[:len(measured)], "batches": batches,
        }
        return measured, report
