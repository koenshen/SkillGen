import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack, closing
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import bird, searchqa
from models import CandidateSkill, SkillAnalysis


class BirdExecution(unittest.TestCase):
    def test_extraction_and_single_read_only_database(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.sqlite"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE t(v)")
                db.executemany("INSERT INTO t VALUES (?)", [("A;B",), ("A;B",), ("D",)])
                db.commit()
            gold = "SELECT DISTINCT v FROM t ORDER BY v ASC"
            response = "reasoning response\n```sql\nSELECT v FROM t ORDER BY v DESC;\nDROP TABLE t;\n```"
            scored = bird.score_response(response, gold, path)
            self.assertEqual((scored["ex"], scored["exec_status"]), (1, "ok"))
            self.assertEqual(scored["predicted_sql"], "SELECT v FROM t ORDER BY v DESC;")
            for sql in ("SELECT 'A;B';", "WITH v(n) AS (SELECT 'A;B') SELECT n FROM v;"):
                self.assertEqual(bird.score_response(sql, "SELECT v FROM t WHERE v='A;B'", path)["ex"], 1)
            self.assertEqual(bird.score_response("SELECT 'a;b'", "SELECT 'A;B'", path)["ex"], 0)
            self.assertEqual(bird.score_response("SELECT 1,2", "SELECT 2,1", path)["ex"], 0)
            other = Path(temp) / "unexpected.sqlite"
            for invalid in ("", "SELECT missing FROM t", "DELETE FROM t",
                            f"ATTACH DATABASE '{other}' AS other", f"VACUUM INTO '{other}'"):
                self.assertEqual(bird.score_response(invalid, gold, path)["exec_status"], "error")
            self.assertFalse(other.exists())
            slow = "WITH RECURSIVE seq(n) AS (VALUES(0) UNION ALL SELECT n+1 FROM seq) SELECT sum(n) FROM seq"
            self.assertEqual(bird.score_response("SELECT 0", slow, path, timeout=0.001)["exec_status"], "timeout")
            self.assertEqual(bird.score_response("SELECT count(*) FROM t", "SELECT 3", path)["ex"], 1)


class NativeFixedFlow(unittest.TestCase):
    def test_three_tasks_train_verify_and_test_with_their_grader(self):
        # The pipeline, verifier, paired evaluator and scorer run normally.
        # Only model-dependent analysis/generation and chat replies are replaced.
        for bench in ("triviaqa", "hotpotqa", "bird"):
            with self.subTest(bench=bench), tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
                root = Path(tmp)
                data = root / "data"
                db_root = root / "db"
                db = bird.database_path(db_root, "tiny")
                db.parent.mkdir(parents=True)
                with closing(sqlite3.connect(db)) as conn:
                    conn.execute('CREATE TABLE t(secret TEXT)')
                    conn.execute("INSERT INTO t VALUES ('A;B')")
                    conn.commit()
                spec = {}
                gold = "SELECT secret FROM t"
                for split in ("train", "val", "test"):
                    ids = [f"{split}-{i}" for i in range(4)]
                    base = [{"id": i, "question": f"item {i}?"} for i in ids]
                    for row in base:
                        if bench == "bird":
                            row.update(SQL=gold, db_id="tiny", prompt="schema / " + "x"*6200 + " / final suffix")
                        else:
                            row.update(context="[DOC] first[DOC] second", answers=["alternative", "Right answer"])
                    dest = data / split / "items.json"
                    dest.parent.mkdir(parents=True)
                    dest.write_text(json.dumps(base))
                    spec[split] = (len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest())
                mocks.enter_context(patch.dict(searchqa.FIXED_SPLITS, {bench: spec}))
                set_inputs = []
                def chat(messages, **kwargs):
                    user = messages[1]["content"]
                    set_inputs.append(user)
                    self.assertNotIn(gold, user)
                    if bench == "bird":
                        self.assertIn(" / final suffix", user)
                    content = "SELECT 'bad'" if bench == "bird" else "<answer>bad</answer>"
                    if "USE-THIS-SKILL" in user:
                        content = "SELECT 'A;B';" if bench == "bird" else "<answer>the right answer</answer>"
                    return SimpleNamespace(content=content, tool_calls=None)
                mocks.enter_context(patch("llm.chat_multi_turn", side_effect=chat))
                mocks.enter_context(patch("pipeline.run_induction", return_value=SkillAnalysis(
                    "analysis", bench, bench, "binary", "", 4, 4, 0)))
                mocks.enter_context(patch("pipeline.generate_skill", return_value=CandidateSkill(
                    "skill", "analysis", "USE-THIS-SKILL", "")))
                mocks.enter_context(patch("llm.chat_json", return_value={}))
                out = root / "result"
                config = Path(searchqa.__file__).resolve().parents[1] / "config.yaml"
                mocks.enter_context(patch.object(sys, "argv", ["fixed", "--benchmark", bench, "--split-dir",
                    str(data), "--db-root", str(db_root), "--config", str(config), "--output", str(out)]))
                searchqa.main()
                summary = json.loads((out / "summary.json").read_text())
                metric = "ex" if bench == "bird" else "em"
                self.assertTrue(summary["skill_deployed"])
                self.assertEqual(summary["baseline"][metric], 0)
                self.assertEqual(summary["final"][metric], 1)
                self.assertEqual(summary["final"]["n"], 4)
                self.assertEqual(len(set_inputs), 20)  # train, baseline dev, candidate dev, both tests
                run, = (out / "artifacts/runs").iterdir()
                verification = json.loads(next(run.glob("verification/*/verification_summary.json")).read_text())
                self.assertTrue(verification["result"]["passed"])
                self.assertEqual(verification["result"]["net_gain"], 4)
                if bench == "bird":
                    for case in verification["result"]["cases"]:
                        self.assertEqual(case["baseline_execution"], "result row sets differ")
                        self.assertEqual(case["skill_execution"], "ok")
                self.assertTrue((out / "test_token_usage.json").is_file())
                ids_file = data / "test/items.json"
                ids_file.write_text(json.dumps(json.loads(ids_file.read_text())[::-1]))
                with self.assertRaisesRegex(ValueError, "release order"):
                    searchqa.load_splits(data, bench)
