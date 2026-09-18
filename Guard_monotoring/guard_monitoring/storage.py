from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    status: str
    progress: float | None
    frames_processed: int
    total_frames: int | None
    source_path: str
    output_dir: str
    error: str | None
    summary: dict | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class SQLiteJobStore:
    """Persistent single-node job state using Python's built-in SQLite.

    This avoids losing all job metadata on a normal process restart. It is suitable for
    a single FastAPI/GPU node. A multi-node deployment still needs a shared queue and
    shared database; this class does not pretend to solve distributed scheduling.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def _initialize(self) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS guard_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL,
                    frames_processed INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER,
                    source_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    error TEXT,
                    summary_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = utc_now()
            con.execute(
                """
                UPDATE guard_jobs
                   SET status='failed',
                       error=COALESCE(error, 'Server restarted before job completed'),
                       updated_at=?
                 WHERE status IN ('queued','processing')
                """,
                (now,),
            )
            con.commit()

    def create(self, job_id: str, source_path: str, output_dir: str) -> JobRecord:
        now = utc_now()
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO guard_jobs (
                    job_id,status,progress,frames_processed,total_frames,
                    source_path,output_dir,error,summary_json,created_at,updated_at
                ) VALUES (?, 'queued', 0, 0, NULL, ?, ?, NULL, NULL, ?, ?)
                """,
                (job_id, source_path, output_dir, now, now),
            )
            con.commit()
        return self.get(job_id)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM guard_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def update(self, job_id: str, **changes) -> JobRecord:
        allowed = {
            "status",
            "progress",
            "frames_processed",
            "total_frames",
            "source_path",
            "output_dir",
            "error",
            "summary",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")

        columns = []
        values = []
        for key, value in changes.items():
            if key == "summary":
                columns.append("summary_json=?")
                values.append(json.dumps(value) if value is not None else None)
            else:
                columns.append(f"{key}=?")
                values.append(value)
        columns.append("updated_at=?")
        values.append(utc_now())
        values.append(job_id)

        with self._lock, self._connect() as con:
            cur = con.execute(
                f"UPDATE guard_jobs SET {', '.join(columns)} WHERE job_id=?",
                tuple(values),
            )
            if cur.rowcount != 1:
                raise KeyError(job_id)
            con.commit()
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def _row(self, row: sqlite3.Row) -> JobRecord:
        summary = json.loads(row["summary_json"]) if row["summary_json"] else None
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            frames_processed=row["frames_processed"],
            total_frames=row["total_frames"],
            source_path=row["source_path"],
            output_dir=row["output_dir"],
            error=row["error"],
            summary=summary,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
