"""BIRD execution accuracy on the fixed, read-only SQLite databases."""
from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


DEFAULT_DB_ROOT = Path("/mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820/data_dev/dev_databases")


def database_path(root: Path, db_id: str) -> Path:
    if not isinstance(db_id, str) or not db_id.strip() or Path(db_id).name != db_id or db_id in (".", ".."):
        raise ValueError(f"Invalid BIRD db_id: {db_id!r}")
    return root / db_id / f"{db_id}.sqlite"


def extract_sql(response: str) -> str:
    """Use the first fenced statement, stopping at an unquoted semicolon."""
    content = (response or "").rsplit(" response", 1)[-1]
    block = re.search(r"```(?:sql|sqlite|mysql|postgresql)?\s*([\s\S]*?)```", content, re.I)
    if block:
        content = block.group(1)
    start = re.search(r"(?im)^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b", content)
    if start:
        content = content[start.start():]
    content = content.strip()
    quote = None
    i = 0
    while i < len(content):
        char = content[i]
        if quote:
            if char == quote:
                if content[i + 1:i + 2] == quote:
                    i += 2
                    continue
                quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == ";":
            return content[:i + 1].strip()
        i += 1
    return content


def _authorize(action, _arg1, _arg2, _db, _source):
    # Prevent ATTACH and VACUUM INTO from opening or creating a second database.
    return sqlite3.SQLITE_DENY if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH) else sqlite3.SQLITE_OK


def score_response(response: str, gold_sql: str, db_path: str | Path, timeout: float = 30.0) -> dict:
    """Compare sets of full result rows. Both statements share one time budget."""
    sql = extract_sql(response)
    result = {"ex": 0, "predicted_sql": sql, "exec_status": "error", "error_message": ""}
    if not sql:
        result["error_message"] = "empty SQL"
        return result
    deadline = time.monotonic() + timeout
    timed_out = False

    def interrupted():
        nonlocal timed_out
        timed_out = time.monotonic() > deadline
        return int(timed_out)

    stage = "connect"
    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.execute("PRAGMA temp_store=MEMORY")
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(_authorize)
            db.set_progress_handler(interrupted, 1000)
            stage = "predicted SQL"
            predicted = set(db.execute(sql))
            stage = "gold SQL"
            gold = set(db.execute(gold_sql))
            match = predicted == gold
        result.update(ex=int(match), exec_status="ok", error_message="" if match else "result row sets differ")
    except sqlite3.Error as exc:
        result.update(exec_status="timeout" if timed_out else "error", error_message=f"{stage}: {exc}")
    return result
