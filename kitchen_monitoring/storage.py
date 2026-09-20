import json
import sqlite3
import uuid

from datetime import (
    datetime,
    timezone,
)

from .config import DB_PATH


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


class KitchenStorage:

    def __init__(self):

        self.db_path = DB_PATH

        self._initialize()


    def _connect(self):

        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        conn.row_factory = (
            sqlite3.Row
        )

        conn.execute(
            "PRAGMA journal_mode=WAL;"
        )

        return conn


    def _initialize(self):

        with self._connect() as conn:

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kitchen_sessions (
                    session_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    camera_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    processed_frames INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    fps REAL DEFAULT 0,
                    output_video TEXT,
                    latest_frame TEXT,
                    summary_json TEXT DEFAULT '{}',
                    error TEXT
                );


                CREATE TABLE IF NOT EXISTS kitchen_violations (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    staff_label TEXT NOT NULL,
                    requirement TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL,
                    started_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                );


                CREATE INDEX IF NOT EXISTS
                idx_kitchen_violation_session
                ON kitchen_violations(session_id);


                CREATE TABLE IF NOT EXISTS kitchen_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    staff_detected INTEGER,
                    fully_compliant INTEGER,
                    open_violations INTEGER,
                    compliance_score REAL
                );
                """
            )


    # ========================================================
    # SESSION
    # ========================================================

    def create_session(
        self,
        session_id,
        source_type,
        source_path,
        camera_id,
    ):

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO kitchen_sessions (
                    session_id,
                    source_type,
                    source_path,
                    camera_id,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    source_type,
                    source_path,
                    camera_id,
                    "queued",
                    utc_now(),
                ),
            )


    def update_session(
        self,
        session_id,
        **fields,
    ):

        if not fields:
            return

        allowed = {
            "status",
            "started_at",
            "ended_at",
            "processed_frames",
            "total_frames",
            "fps",
            "output_video",
            "latest_frame",
            "summary_json",
            "error",
        }

        values = {
            k: v
            for k, v in fields.items()
            if k in allowed
        }

        if "summary_json" in values:
            if not isinstance(
                values["summary_json"],
                str,
            ):
                values[
                    "summary_json"
                ] = json.dumps(
                    values[
                        "summary_json"
                    ]
                )

        if not values:
            return

        sql = (
            "UPDATE kitchen_sessions SET "
            +
            ", ".join(
                f"{key} = ?"
                for key in values
            )
            +
            " WHERE session_id = ?"
        )

        parameters = (
            list(values.values())
            +
            [session_id]
        )

        with self._connect() as conn:

            conn.execute(
                sql,
                parameters,
            )


    def get_session(
        self,
        session_id,
    ):

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM kitchen_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        data = dict(row)

        try:
            data["summary"] = (
                json.loads(
                    data.pop(
                        "summary_json",
                        "{}",
                    )
                    or "{}"
                )
            )
        except Exception:
            data["summary"] = {}

        return data


    # ========================================================
    # VIOLATIONS
    # ========================================================

    def touch_violation(
        self,
        session_id,
        track_id,
        staff_label,
        requirement,
        violation_type,
        severity,
        confidence,
    ):

        now = utc_now()

        with self._connect() as conn:

            existing = conn.execute(
                """
                SELECT *
                FROM kitchen_violations
                WHERE session_id = ?
                AND track_id = ?
                AND requirement = ?
                AND active = 1
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (
                    session_id,
                    track_id,
                    requirement,
                ),
            ).fetchone()


            if existing:

                conn.execute(
                    """
                    UPDATE kitchen_violations
                    SET
                        last_seen_at = ?,
                        confidence = ?,
                        violation_type = ?,
                        severity = ?
                    WHERE event_id = ?
                    """,
                    (
                        now,
                        confidence,
                        violation_type,
                        severity,
                        existing["event_id"],
                    ),
                )

                return existing[
                    "event_id"
                ]


            event_id = (
                "KV-"
                +
                uuid.uuid4().hex[:12]
            )

            conn.execute(
                """
                INSERT INTO kitchen_violations (
                    event_id,
                    session_id,
                    track_id,
                    staff_label,
                    requirement,
                    violation_type,
                    severity,
                    confidence,
                    started_at,
                    last_seen_at,
                    active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    event_id,
                    session_id,
                    track_id,
                    staff_label,
                    requirement,
                    violation_type,
                    severity,
                    confidence,
                    now,
                    now,
                ),
            )

        return event_id


    def close_violation(
        self,
        session_id,
        track_id,
        requirement,
    ):

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE kitchen_violations
                SET
                    active = 0,
                    last_seen_at = ?
                WHERE session_id = ?
                AND track_id = ?
                AND requirement = ?
                AND active = 1
                """,
                (
                    utc_now(),
                    session_id,
                    track_id,
                    requirement,
                ),
            )


    def close_track_violations(
        self,
        session_id,
        track_id,
    ):

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE kitchen_violations
                SET
                    active = 0,
                    last_seen_at = ?
                WHERE session_id = ?
                AND track_id = ?
                AND active = 1
                """,
                (
                    utc_now(),
                    session_id,
                    track_id,
                ),
            )


    def close_all_violations(
        self,
        session_id,
    ):

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE kitchen_violations
                SET
                    active = 0,
                    last_seen_at = ?
                WHERE session_id = ?
                AND active = 1
                """,
                (
                    utc_now(),
                    session_id,
                ),
            )


    def list_violations(
        self,
        session_id,
        active_only=False,
    ):

        sql = """
            SELECT *
            FROM kitchen_violations
            WHERE session_id = ?
        """

        params = [
            session_id
        ]

        if active_only:

            sql += (
                " AND active = 1"
            )

        sql += (
            " ORDER BY started_at DESC"
        )

        with self._connect() as conn:

            rows = conn.execute(
                sql,
                params,
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


    # ========================================================
    # TREND
    # ========================================================

    def add_metric(
        self,
        session_id,
        summary,
    ):

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO kitchen_metrics (
                    session_id,
                    timestamp,
                    staff_detected,
                    fully_compliant,
                    open_violations,
                    compliance_score
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    utc_now(),
                    summary.get(
                        "staff_detected",
                        0,
                    ),
                    summary.get(
                        "fully_compliant",
                        0,
                    ),
                    summary.get(
                        "open_violations",
                        0,
                    ),
                    summary.get(
                        "compliance_score"
                    ),
                ),
            )


    def get_metrics(
        self,
        session_id,
        limit=100,
    ):

        with self._connect() as conn:

            rows = conn.execute(
                """
                SELECT *
                FROM kitchen_metrics
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    session_id,
                    int(limit),
                ),
            ).fetchall()

        values = [
            dict(row)
            for row in rows
        ]

        values.reverse()

        return values


STORE = KitchenStorage()