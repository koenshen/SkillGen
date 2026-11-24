"""Offline coverage of SearchQA candidate validation and native handoff."""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from benchmarks import searchqa
from benchmarks.searchqa_dev import DevEarlyStop
from effectiveness import verify_effectiveness
from models import CandidateSkill, SkillAnalysis, SkillItem, TaskInstance, TaskType, Trajectory


def trace(i, passed, *, error=False):
    return Trajectory(i, i, {}, [], "yes" if passed else "no", success=passed,
                      score=float(passed), metadata={"agent_exception": True} if error else {})


class DevBatches(unittest.TestCase):
    def setUp(self):
        self.items = [TaskInstance(str(i), "") for i in range(8)]
        self.baseline = {str(i): trace(str(i), i % 2 == 0) for i in range(8)}

    def test_order_equality_and_observed_history(self):
        verifier = DevEarlyStop("easy-first", 2, True)
        verifier.seed(self.items, self.baseline)
        calls = []

        def run(batch):
            calls.append([i.instance_id for i in batch])
            return [trace(i.instance_id, True) for i in reversed(batch)]

        _, result = verifier.evaluate(list(reversed(self.items)), 2, run)
        self.assertEqual(calls, [["0", "2"], ["4", "6"], ["1", "3"]])
        self.assertEqual((result["correct_lower"], result["correct_upper"]), (6, 8))
        self.assertTrue(result["stopped"])
        self.assertEqual(len(verifier.history["7"]), 1)
        self.assertEqual(verifier.history["3"], [False, True])
        # Across versions, a repaired (False -> True) item precedes a steady
        # failure on easy-first and follows a consistently correct item.
        calls.clear()
        verifier.evaluate(self.items, 10, run)
        self.assertEqual(calls[2:], [["1", "3"], ["5", "7"]])

    def test_complete_feedback_even_after_rejection_or_when_accept_disabled(self):
        for passed in (False, True):
            verifier = DevEarlyStop(batch_size=2)
            verifier.seed(self.items, self.baseline)
            rows, result = verifier.evaluate(self.items, 1, lambda batch: [
                trace(i.instance_id, passed) for i in batch])
            self.assertFalse(result["stopped"])
            self.assertEqual(len(rows), 8)
            self.assertEqual(result["evaluated_ids"][:4], ["1", "3", "5", "7"])
            self.assertEqual(result["reject_certified_at"], None if passed else 4)

    def test_one_failed_batch_does_not_abort_other_batches(self):
        verifier = DevEarlyStop(batch_size=4, early_accept=True)
        verifier.seed(self.items, self.baseline)
        with self.assertRaisesRegex(RuntimeError, "Every SearchQA rollout failed"):
            verifier.evaluate(self.items, 1, lambda batch: [trace(i.instance_id, False, error=True) for i in batch])
        self.assertTrue(all(len(v) == 1 for v in verifier.history.values()))
        rows, result = verifier.evaluate(self.items, 1, lambda batch: [
            trace(i.instance_id, False, error=int(i.instance_id) % 2 == 1) for i in batch])
        self.assertEqual(len(rows), 8)
        self.assertFalse(result["stopped"])

    def test_missing_or_duplicate_result_cannot_certify_a_decision(self):
        verifier = DevEarlyStop()
        verifier.seed(self.items, self.baseline)
        for run in (lambda b: [], lambda b: [trace(b[0].instance_id, True)]*len(b)):
            with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
                verifier.evaluate(self.items, 1, run)
        self.assertTrue(all(len(v) == 1 for v in verifier.history.values()))

    def test_default_native_order_and_batch_level_errors(self):
        for i in self.items:
            i.metadata = {"benchmark": "searchqa"}
            i.ground_truth = ["yes"]
        failures = self.items[1::2]
        successes = self.items[::2]
        skill = SkillItem("s", "body", "")
        verifier = DevEarlyStop(batch_size=4, early_accept=True)
        verifier.seed(self.items, self.baseline)

        def agent(instance, **kwargs):
            if not self.baseline[instance.instance_id].success:
                raise RuntimeError("simulated HTTP failure")
            return trace(instance.instance_id, True)

        with patch("trajectory.run_agent", side_effect=agent):
            full, _ = verify_effectiveness(skill, failures, successes, TaskType.BINARY,
                                           baseline_cache=self.baseline)
            ranked, _ = verify_effectiveness(skill, failures, successes, TaskType.BINARY,
                                             baseline_cache=self.baseline, dev_early_stop=verifier)
        self.assertIsNone(full.dev_evaluation)
        self.assertEqual((full.net_gain, ranked.net_gain), (0, 0))
        self.assertEqual(full.cases, ranked.cases)
        self.assertEqual(ranked.dev_evaluation["n_evaluated"], 8)


class NativeSearchQAFlow(unittest.TestCase):
    def test_full_failed_feedback_then_certified_pass_and_full_test(self):
        import pipeline
        import yaml

        def rows(name, n):
            return [dict(id=f"{name}{i}", question="?", context="yes", answers=["yes"]) for i in range(n)]

        data = {"train": rows("train", 2), "val": rows("dev", 12), "test": rows("test", 9)}
        analysis = SkillAnalysis("analysis", "searchqa", "SearchQA", "binary", "", 2, 2, 0)
        bad = CandidateSkill("bad", "analysis", "first candidate", "")
        good = CandidateSkill("good", "analysis", "next candidate", "")

        def agent(instance, skill_bundle=None, config=None):
            yes = (skill_bundle is not None and skill_bundle.body == good.body) or (
                skill_bundle is None and instance.instance_id in {"dev0", "dev2", "dev4"})
            return trace(instance.instance_id, yes)

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
            root = Path(tmp)
            cfg = yaml.safe_load((Path(pipeline.__file__).with_name("config.yaml")).read_text())
            cfg["pipeline"]["artifact_root"] = str(root/"runs")
            cfg["skill_output"]["path"] = str(root/"skills")
            cfg["generation"]["candidate_output_dir"] = str(root/"candidates")
            config = root/"config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            output = root/"test"
            argv = ["searchqa", "--output", str(output), "--config", str(config),
                    "--dev-early-stop", "--dev-early-accept", "--dev-order", "easy-first"]
            mocks.enter_context(patch.object(sys, "argv", argv))
            mocks.enter_context(patch.object(searchqa, "load_splits", return_value=data))
            mocks.enter_context(patch("trajectory.run_agent", side_effect=agent))
            mocks.enter_context(patch("pipeline.run_induction", return_value=analysis))
            mocks.enter_context(patch("pipeline.generate_skill", return_value=bad))
            refine = mocks.enter_context(patch("pipeline.refine_skill", return_value=good))
            mocks.enter_context(patch("llm.chat_json", return_value={}))
            searchqa.main()
            refine.assert_called_once()
            run_dir, = (root/"runs").iterdir()
            failed = json.loads((run_dir/"verification/round_1/verification_summary.json").read_text())
            accepted = json.loads((run_dir/"verification/round_2/verification_summary.json").read_text())
            self.assertEqual(failed["result"]["paired_n"], 12)
            self.assertFalse(failed["result"]["passed"])
            self.assertEqual(failed["result"]["net_gain"], -3)
            evaluated = accepted["dev_evaluation"]
            self.assertEqual((evaluated["n_evaluated"], evaluated["n_total"]), (8, 12))
            self.assertTrue(accepted["result"]["passed"])
            self.assertIsNone(accepted["result"]["skill_acc"])
            self.assertIsNone(accepted["result"]["net_gain"])
            accepted_dir = run_dir/"verification/round_2"
            cases = json.loads((accepted_dir/"verification_case_analyses.json").read_text())["case_analyses"]
            self.assertEqual({c["instance_id"] for c in cases},
                             {c["instance_id"] for c in accepted["result"]["cases"]})
            self.assertEqual(len(cases), 5)
            summary = json.loads((output/"summary.json").read_text())
            self.assertTrue(summary["skill_deployed"])
            self.assertEqual((summary["baseline"]["n"], summary["final"]["n"]), (9, 9))
            self.assertEqual(summary["final"]["em"], 1.0)
            saved = json.loads(next((root/"skills").glob(f"*/{summary['skill_id']}.json")).read_text())
            self.assertEqual(saved["body"], good.body)
            self.assertIsNone(saved["verification_history"][0]["net_gain"])


if __name__ == "__main__":
    unittest.main()
