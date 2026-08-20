from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.review_models import ContractReviewRun


class ReviewRunTransitionError(RuntimeError):
    pass


class ContractReviewRepository:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_runs (
                    run_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'ready', 'failed')
                    ),
                    review_rule_text TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    FOREIGN KEY (contract_id)
                        REFERENCES contracts(contract_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_review_runs_contract
                    ON review_runs(contract_id, created_at DESC);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _run_from_row(row: sqlite3.Row | None) -> ContractReviewRun | None:
        if row is None:
            return None
        return ContractReviewRun(
            run_id=row["run_id"],
            contract_id=row["contract_id"],
            status=row["status"],
            review_rule_text=row["review_rule_text"],
            result=json.loads(row["result_json"]),
            progress=json.loads(row["progress_json"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )

    def create_run(
        self,
        contract_id: str,
        review_rule_text: str,
    ) -> ContractReviewRun:
        run_id = str(uuid.uuid4())
        timestamp = self._now()
        progress = {"stage": "queued", "message": "等待开始风险审查"}
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO review_runs (
                    run_id, contract_id, status, review_rule_text,
                    result_json, progress_json, created_at
                ) VALUES (?, ?, 'queued', ?, '{}', ?, ?)
                """,
                (
                    run_id,
                    contract_id,
                    review_rule_text,
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                ),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> ContractReviewRun | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM review_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._run_from_row(row)

    def mark_processing(
        self,
        run_id: str,
        progress: dict[str, Any],
    ) -> ContractReviewRun:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE review_runs
                SET status = 'processing', started_at = ?,
                    progress_json = ?, error_message = NULL
                WHERE run_id = ? AND status = 'queued'
                """,
                (
                    timestamp,
                    json.dumps(progress, ensure_ascii=False),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewRunTransitionError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def update_progress(
        self,
        run_id: str,
        progress: dict[str, Any],
    ) -> ContractReviewRun:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE review_runs
                SET progress_json = ?
                WHERE run_id = ? AND status = 'processing'
                """,
                (json.dumps(progress, ensure_ascii=False), run_id),
            )
            if cursor.rowcount != 1:
                raise ReviewRunTransitionError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def mark_ready(
        self,
        run_id: str,
        result: dict[str, Any],
        progress: dict[str, Any],
    ) -> ContractReviewRun:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE review_runs
                SET status = 'ready', result_json = ?, progress_json = ?,
                    completed_at = ?, error_message = NULL
                WHERE run_id = ? AND status = 'processing'
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewRunTransitionError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def mark_failed(
        self,
        run_id: str,
        error_message: str,
    ) -> ContractReviewRun:
        timestamp = self._now()
        progress = {"stage": "failed", "message": "风险审查失败"}
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE review_runs
                SET status = 'failed', progress_json = ?, completed_at = ?,
                    error_message = ?
                WHERE run_id = ? AND status = 'processing'
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                    error_message,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewRunTransitionError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def recover_incomplete_runs(self, reason: str) -> int:
        timestamp = self._now()
        progress = {"stage": "failed", "message": "风险审查失败"}
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE review_runs
                SET status = 'failed', progress_json = ?, completed_at = ?,
                    error_message = ?
                WHERE status IN ('queued', 'processing')
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                    reason,
                ),
            )
        return cursor.rowcount
