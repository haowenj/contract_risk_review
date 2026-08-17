from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation_models import EvaluationCase, EvaluationRun, EvaluationRunItem


RUN_SCOPES = frozenset({"single", "all"})
RUN_STATUSES = frozenset({"queued", "processing", "ready", "failed"})


class EvaluationRepository:
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
                CREATE TABLE IF NOT EXISTS evaluation_cases (
                    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    question TEXT NOT NULL,
                    expected_source_object_indices TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (contract_id)
                        REFERENCES contracts(contract_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evaluation_cases_contract
                    ON evaluation_cases(contract_id, sort_order);

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK (scope IN ('single', 'all')),
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'ready', 'failed')
                    ),
                    index_version TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    config_snapshot TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    FOREIGN KEY (contract_id)
                        REFERENCES contracts(contract_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_contract
                    ON evaluation_runs(contract_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS evaluation_run_items (
                    run_id TEXT NOT NULL,
                    case_id INTEGER NOT NULL,
                    sort_order INTEGER NOT NULL,
                    question_snapshot TEXT NOT NULL,
                    expected_source_object_indices_snapshot TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, case_id),
                    FOREIGN KEY (run_id)
                        REFERENCES evaluation_runs(run_id)
                        ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _case_from_row(row: sqlite3.Row | None) -> EvaluationCase | None:
        if row is None:
            return None
        return EvaluationCase(
            case_id=row["case_id"],
            contract_id=row["contract_id"],
            index_version=row["index_version"],
            question=row["question"],
            expected_source_object_indices=json.loads(
                row["expected_source_object_indices"]
            ),
            sort_order=row["sort_order"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row | None) -> EvaluationRun | None:
        if row is None:
            return None
        config_snapshot = json.loads(row["config_snapshot"])
        return EvaluationRun(
            run_id=row["run_id"],
            contract_id=row["contract_id"],
            scope=row["scope"],
            status=row["status"],
            index_version=row["index_version"],
            pipeline_version=row["pipeline_version"],
            config_snapshot=config_snapshot,
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> EvaluationRunItem:
        return EvaluationRunItem(
            run_id=row["run_id"],
            case_id=row["case_id"],
            question_snapshot=row["question_snapshot"],
            expected_source_object_indices_snapshot=json.loads(
                row["expected_source_object_indices_snapshot"]
            ),
            result=json.loads(row["result_json"]),
        )

    def list_cases(self, contract_id: str) -> list[EvaluationCase]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evaluation_cases
                WHERE contract_id = ?
                ORDER BY sort_order ASC, case_id ASC
                """,
                (contract_id,),
            ).fetchall()
        return [case for row in rows if (case := self._case_from_row(row))]

    def get_case(self, contract_id: str, case_id: int) -> EvaluationCase | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM evaluation_cases
                WHERE contract_id = ? AND case_id = ?
                """,
                (contract_id, case_id),
            ).fetchone()
        return self._case_from_row(row)

    def replace_cases(
        self,
        contract_id: str,
        index_version: str,
        entries: list[tuple[str, list[int]]],
    ) -> list[EvaluationCase]:
        normalized_entries: list[tuple[str, list[int]]] = []
        for question, expected_indices in entries:
            normalized_question = question.strip()
            if not normalized_question:
                raise ValueError("evaluation question must not be empty")
            if any(type(index) is not int for index in expected_indices):
                raise ValueError("expected source object indices must be integers")
            normalized_entries.append(
                (normalized_question, list(dict.fromkeys(expected_indices)))
            )

        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            connection.execute(
                "DELETE FROM evaluation_cases WHERE contract_id = ?",
                (contract_id,),
            )
            for sort_order, (question, expected_indices) in enumerate(
                normalized_entries
            ):
                connection.execute(
                    """
                    INSERT INTO evaluation_cases (
                        contract_id, index_version, question,
                        expected_source_object_indices, sort_order,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contract_id,
                        index_version,
                        question,
                        json.dumps(expected_indices),
                        sort_order,
                        timestamp,
                        timestamp,
                    ),
                )
        return self.list_cases(contract_id)

    def create_run(
        self,
        contract_id: str,
        scope: str,
        index_version: str,
        config_snapshot: dict[str, Any],
        cases: list[EvaluationCase],
    ) -> EvaluationRun:
        if scope not in RUN_SCOPES:
            raise ValueError(f"unsupported evaluation scope: {scope}")
        run_id = str(uuid.uuid4())
        timestamp = self._now()
        pipeline_version = str(config_snapshot.get("pipeline_version", ""))
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, contract_id, scope, status, index_version,
                    pipeline_version, config_snapshot, created_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    contract_id,
                    scope,
                    index_version,
                    pipeline_version,
                    json.dumps(config_snapshot, ensure_ascii=False),
                    timestamp,
                ),
            )
            connection.executemany(
                """
                INSERT INTO evaluation_run_items (
                    run_id, case_id, sort_order, question_snapshot,
                    expected_source_object_indices_snapshot, result_json
                ) VALUES (?, ?, ?, ?, ?, '{}')
                """,
                [
                    (
                        run_id,
                        case.case_id,
                        case.sort_order,
                        case.question,
                        json.dumps(
                            case.expected_source_object_indices,
                            ensure_ascii=False,
                        ),
                    )
                    for case in cases
                ],
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def mark_processing(self, run_id: str) -> EvaluationRun:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'processing', started_at = ?, error_message = NULL
                WHERE run_id = ?
                """,
                (timestamp, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def save_item(
        self,
        run_id: str,
        case: EvaluationCase,
        result: dict[str, Any],
    ) -> EvaluationRunItem:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_run_items
                SET result_json = ?
                WHERE run_id = ? AND case_id = ?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    run_id,
                    case.case_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError((run_id, case.case_id))
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM evaluation_run_items
                WHERE run_id = ? AND case_id = ?
                """,
                (run_id, case.case_id),
            ).fetchone()
        if row is None:
            raise KeyError((run_id, case.case_id))
        return self._item_from_row(row)

    def mark_ready(self, run_id: str) -> EvaluationRun:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'ready', completed_at = ?, error_message = NULL
                WHERE run_id = ?
                """,
                (timestamp, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def mark_failed(self, run_id: str, error_message: str) -> EvaluationRun:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'failed', completed_at = ?, error_message = ?
                WHERE run_id = ?
                """,
                (timestamp, error_message, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)
        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> EvaluationRun | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._run_from_row(row)

    def list_run_items(self, run_id: str) -> list[EvaluationRunItem]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evaluation_run_items
                WHERE run_id = ?
                ORDER BY sort_order ASC, case_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def latest_run(self, contract_id: str) -> EvaluationRun | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM evaluation_runs
                WHERE contract_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (contract_id,),
            ).fetchone()
        return self._run_from_row(row)

    def recover_incomplete_runs(self, reason: str) -> int:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'failed', completed_at = ?, error_message = ?
                WHERE status IN ('queued', 'processing')
                """,
                (timestamp, reason),
            )
        return cursor.rowcount
