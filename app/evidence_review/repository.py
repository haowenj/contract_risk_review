from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evidence_review.schemas import HumanDecision


class RuleSetTransitionError(RuntimeError):
    pass


class EvidenceRunTransitionError(RuntimeError):
    pass


class DecisionConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuleSetRecord:
    rule_set_id: str
    name: str
    version: int
    status: str
    source_filename: str
    source_path: str
    source_sha256: str
    parsed_rules: dict[str, Any]
    created_at: str
    updated_at: str
    activated_at: str | None
    error_message: str | None


@dataclass(frozen=True)
class EvidenceReviewRunRecord:
    run_id: str
    contract_id: str
    rule_set_id: str
    status: str
    human_status: str
    rule_snapshot: dict[str, Any]
    progress: dict[str, Any]
    created_at: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None


@dataclass(frozen=True)
class EvidenceReviewItemRecord:
    run_id: str
    rule_item_id: str
    ordinal: int
    rule_snapshot: dict[str, Any]
    evidence_package: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class EvidenceReviewDecisionRecord:
    run_id: str
    rule_item_id: str
    decision: HumanDecision
    updated_at: str


class EvidenceReviewRepository:
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
                CREATE TABLE IF NOT EXISTS review_rule_sets (
                    rule_set_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'draft', 'active', 'failed')
                    ),
                    source_filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    parsed_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    activated_at TEXT,
                    error_message TEXT,
                    UNIQUE(name, version)
                );

                CREATE INDEX IF NOT EXISTS idx_review_rule_sets_status
                    ON review_rule_sets(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS evidence_review_runs (
                    run_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    rule_set_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'ready', 'failed')
                    ),
                    human_status TEXT NOT NULL CHECK (
                        human_status IN ('pending', 'in_progress', 'completed')
                    ),
                    rule_snapshot_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_review_runs_contract
                    ON evidence_review_runs(contract_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS evidence_review_items (
                    run_id TEXT NOT NULL,
                    rule_item_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                    rule_snapshot_json TEXT NOT NULL,
                    evidence_package_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, rule_item_id),
                    FOREIGN KEY (run_id)
                        REFERENCES evidence_review_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_review_items_order
                    ON evidence_review_items(run_id, ordinal);

                CREATE TABLE IF NOT EXISTS evidence_review_decisions (
                    run_id TEXT NOT NULL,
                    rule_item_id TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, rule_item_id),
                    FOREIGN KEY (run_id, rule_item_id)
                        REFERENCES evidence_review_items(run_id, rule_item_id)
                        ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> RuleSetRecord | None:
        if row is None:
            return None
        return RuleSetRecord(
            rule_set_id=row["rule_set_id"],
            name=row["name"],
            version=row["version"],
            status=row["status"],
            source_filename=row["source_filename"],
            source_path=row["source_path"],
            source_sha256=row["source_sha256"],
            parsed_rules=json.loads(row["parsed_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            activated_at=row["activated_at"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _evidence_run_from_row(
        row: sqlite3.Row | None,
    ) -> EvidenceReviewRunRecord | None:
        if row is None:
            return None
        return EvidenceReviewRunRecord(
            run_id=row["run_id"],
            contract_id=row["contract_id"],
            rule_set_id=row["rule_set_id"],
            status=row["status"],
            human_status=row["human_status"],
            rule_snapshot=json.loads(row["rule_snapshot_json"]),
            progress=json.loads(row["progress_json"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _evidence_item_from_row(
        row: sqlite3.Row,
    ) -> EvidenceReviewItemRecord:
        return EvidenceReviewItemRecord(
            run_id=row["run_id"],
            rule_item_id=row["rule_item_id"],
            ordinal=row["ordinal"],
            rule_snapshot=json.loads(row["rule_snapshot_json"]),
            evidence_package=json.loads(row["evidence_package_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _decision_from_row(
        row: sqlite3.Row | None,
    ) -> EvidenceReviewDecisionRecord | None:
        if row is None:
            return None
        return EvidenceReviewDecisionRecord(
            run_id=row["run_id"],
            rule_item_id=row["rule_item_id"],
            decision=HumanDecision.model_validate_json(row["decision_json"]),
            updated_at=row["updated_at"],
        )

    def create_rule_set(
        self,
        *,
        name: str,
        source_filename: str,
        source_path: Path | str,
        source_sha256: str,
    ) -> RuleSetRecord:
        timestamp = self._now()
        rule_set_id = str(uuid.uuid4())
        with self._write_lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM review_rule_sets WHERE name = ?",
                (name.strip(),),
            ).fetchone()
            version = int(row[0])
            connection.execute(
                """
                INSERT INTO review_rule_sets (
                    rule_set_id, name, version, status, source_filename,
                    source_path, source_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    rule_set_id,
                    name.strip(),
                    version,
                    Path(source_filename).name,
                    str(source_path),
                    source_sha256,
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_rule_set(rule_set_id)  # type: ignore[return-value]

    def get_rule_set(self, rule_set_id: str) -> RuleSetRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM review_rule_sets WHERE rule_set_id = ?",
                (rule_set_id,),
            ).fetchone()
        return self._from_row(row)

    def list_rule_sets(self) -> list[RuleSetRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM review_rule_sets ORDER BY created_at DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]  # type: ignore[misc]

    def _transition(
        self,
        rule_set_id: str,
        *,
        expected: str,
        target: str,
        parsed_rules: dict[str, Any] | None = None,
        error_message: str | None = None,
        activated: bool = False,
    ) -> RuleSetRecord:
        timestamp = self._now()
        assignments = ["status = ?", "updated_at = ?", "error_message = ?"]
        values: list[Any] = [target, timestamp, error_message]
        if parsed_rules is not None:
            assignments.append("parsed_json = ?")
            values.append(json.dumps(parsed_rules, ensure_ascii=False))
        if activated:
            assignments.append("activated_at = ?")
            values.append(timestamp)
        values.extend([rule_set_id, expected])
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE review_rule_sets SET {', '.join(assignments)} "
                "WHERE rule_set_id = ? AND status = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise RuleSetTransitionError(rule_set_id)
        return self.get_rule_set(rule_set_id)  # type: ignore[return-value]

    def mark_rule_set_processing(self, rule_set_id: str) -> RuleSetRecord:
        return self._transition(
            rule_set_id, expected="queued", target="processing"
        )

    def mark_rule_set_draft(
        self, rule_set_id: str, parsed_rules: dict[str, Any]
    ) -> RuleSetRecord:
        return self._transition(
            rule_set_id,
            expected="processing",
            target="draft",
            parsed_rules=parsed_rules,
        )

    def activate_rule_set(self, rule_set_id: str) -> RuleSetRecord:
        return self._transition(
            rule_set_id,
            expected="draft",
            target="active",
            activated=True,
        )

    def mark_rule_set_failed(
        self, rule_set_id: str, error_message: str
    ) -> RuleSetRecord:
        current = self.get_rule_set(rule_set_id)
        if current is None or current.status not in {"queued", "processing"}:
            raise RuleSetTransitionError(rule_set_id)
        return self._transition(
            rule_set_id,
            expected=current.status,
            target="failed",
            error_message=error_message,
        )

    def create_evidence_run(
        self,
        *,
        contract_id: str,
        rule_set_id: str,
        rule_snapshot: dict[str, Any],
    ) -> EvidenceReviewRunRecord:
        run_id = str(uuid.uuid4())
        timestamp = self._now()
        total = len(rule_snapshot.get("review_items", []))
        progress = {
            "stage": "queued",
            "message": "等待开始合同取证",
            "completed": 0,
            "total": total,
        }
        snapshot_json = json.dumps(rule_snapshot, ensure_ascii=False)
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evidence_review_runs (
                    run_id, contract_id, rule_set_id, status, human_status,
                    rule_snapshot_json, progress_json, created_at
                ) VALUES (?, ?, ?, 'queued', 'pending', ?, ?, ?)
                """,
                (
                    run_id,
                    contract_id,
                    rule_set_id,
                    snapshot_json,
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                ),
            )
        return self.get_evidence_run(run_id)  # type: ignore[return-value]

    def get_evidence_run(
        self,
        run_id: str,
    ) -> EvidenceReviewRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_review_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._evidence_run_from_row(row)

    def list_evidence_runs(
        self,
        contract_id: str,
    ) -> list[EvidenceReviewRunRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_review_runs
                WHERE contract_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 20
                """,
                (contract_id,),
            ).fetchall()
        return [self._evidence_run_from_row(row) for row in rows]

    def mark_evidence_run_processing(
        self,
        run_id: str,
    ) -> EvidenceReviewRunRecord:
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            row = connection.execute(
                """
                SELECT rule_snapshot_json FROM evidence_review_runs
                WHERE run_id = ? AND status = 'queued'
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise EvidenceRunTransitionError(run_id)
            snapshot = json.loads(row["rule_snapshot_json"])
            total = len(snapshot.get("review_items", []))
            cursor = connection.execute(
                """
                UPDATE evidence_review_runs
                SET status = 'processing', started_at = ?,
                    progress_json = ?, error_message = NULL
                WHERE run_id = ? AND status = 'queued'
                """,
                (
                    timestamp,
                    json.dumps(
                        {
                            "stage": "processing",
                            "message": "正在提取合同证据",
                            "completed": 0,
                            "total": total,
                        },
                        ensure_ascii=False,
                    ),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise EvidenceRunTransitionError(run_id)
        return self.get_evidence_run(run_id)  # type: ignore[return-value]

    def update_evidence_run_progress(
        self,
        run_id: str,
        progress: dict[str, Any],
    ) -> EvidenceReviewRunRecord:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_review_runs
                SET progress_json = ?
                WHERE run_id = ? AND status = 'processing'
                """,
                (json.dumps(progress, ensure_ascii=False), run_id),
            )
            if cursor.rowcount != 1:
                raise EvidenceRunTransitionError(run_id)
        return self.get_evidence_run(run_id)  # type: ignore[return-value]

    def complete_evidence_run(
        self,
        run_id: str,
        items: list[dict[str, Any]],
        *,
        progress: dict[str, Any],
    ) -> EvidenceReviewRunRecord:
        item_ids = [str(value.get("rule_item_id", "")) for value in items]
        if any(not value for value in item_ids):
            raise ValueError("rule item ids must not be empty")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("rule item ids must be unique")

        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            status_row = connection.execute(
                "SELECT status FROM evidence_review_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if status_row is None or status_row["status"] != "processing":
                raise EvidenceRunTransitionError(run_id)
            for ordinal, value in enumerate(items):
                connection.execute(
                    """
                    INSERT INTO evidence_review_items (
                        run_id, rule_item_id, ordinal, rule_snapshot_json,
                        evidence_package_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item_ids[ordinal],
                        ordinal,
                        json.dumps(
                            value["rule_snapshot"], ensure_ascii=False
                        ),
                        json.dumps(
                            value["evidence_package"], ensure_ascii=False
                        ),
                        timestamp,
                    ),
                )
                raw_package = value["evidence_package"]
                research_package = raw_package.get("research_package")
                research_status = (
                    research_package.get("research_status", "pending")
                    if isinstance(research_package, dict)
                    else "not_required"
                )
                initial_decision = HumanDecision(
                    human_review_status="pending",
                    decision=None,
                    risk_level=None,
                    opinion="",
                    research_status=research_status,
                    research_notes="",
                    updated_at=timestamp,
                )
                connection.execute(
                    """
                    INSERT INTO evidence_review_decisions (
                        run_id, rule_item_id, decision_json, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item_ids[ordinal],
                        initial_decision.model_dump_json(),
                        timestamp,
                    ),
                )
            cursor = connection.execute(
                """
                UPDATE evidence_review_runs
                SET status = 'ready', progress_json = ?, completed_at = ?,
                    error_message = NULL
                WHERE run_id = ? AND status = 'processing'
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise EvidenceRunTransitionError(run_id)
        return self.get_evidence_run(run_id)  # type: ignore[return-value]

    def list_evidence_items(
        self,
        run_id: str,
    ) -> list[EvidenceReviewItemRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_review_items
                WHERE run_id = ? ORDER BY ordinal
                """,
                (run_id,),
            ).fetchall()
        return [self._evidence_item_from_row(row) for row in rows]

    def get_evidence_decision(
        self,
        run_id: str,
        rule_item_id: str,
    ) -> EvidenceReviewDecisionRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM evidence_review_decisions
                WHERE run_id = ? AND rule_item_id = ?
                """,
                (run_id, rule_item_id),
            ).fetchone()
        return self._decision_from_row(row)

    def list_evidence_decisions(
        self,
        run_id: str,
    ) -> list[EvidenceReviewDecisionRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT decision.*
                FROM evidence_review_decisions AS decision
                JOIN evidence_review_items AS item
                  ON item.run_id = decision.run_id
                 AND item.rule_item_id = decision.rule_item_id
                WHERE decision.run_id = ?
                ORDER BY item.ordinal
                """,
                (run_id,),
            ).fetchall()
        return [
            value
            for row in rows
            if (value := self._decision_from_row(row)) is not None
        ]

    def save_evidence_decision(
        self,
        run_id: str,
        rule_item_id: str,
        decision: HumanDecision,
        expected_updated_at: str,
    ) -> EvidenceReviewDecisionRecord:
        updated_at = decision.updated_at.isoformat()
        with self._write_lock, self._connection() as connection:
            run = connection.execute(
                "SELECT status FROM evidence_review_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            if run["status"] != "ready":
                raise EvidenceRunTransitionError(run_id)
            cursor = connection.execute(
                """
                UPDATE evidence_review_decisions
                SET decision_json = ?, updated_at = ?
                WHERE run_id = ? AND rule_item_id = ? AND updated_at = ?
                """,
                (
                    decision.model_dump_json(),
                    updated_at,
                    run_id,
                    rule_item_id,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount != 1:
                existing = connection.execute(
                    """
                    SELECT 1 FROM evidence_review_decisions
                    WHERE run_id = ? AND rule_item_id = ?
                    """,
                    (run_id, rule_item_id),
                ).fetchone()
                if existing is None:
                    raise KeyError(rule_item_id)
                raise DecisionConflictError(rule_item_id)

            rows = connection.execute(
                """
                SELECT decision_json FROM evidence_review_decisions
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
            completed = sum(
                HumanDecision.model_validate_json(row["decision_json"])
                .human_review_status
                == "completed"
                for row in rows
            )
            if completed == 0:
                human_status = "pending"
            elif completed == len(rows):
                human_status = "completed"
            else:
                human_status = "in_progress"
            connection.execute(
                """
                UPDATE evidence_review_runs SET human_status = ?
                WHERE run_id = ?
                """,
                (human_status, run_id),
            )
        return self.get_evidence_decision(  # type: ignore[return-value]
            run_id,
            rule_item_id,
        )

    def mark_evidence_run_failed(
        self,
        run_id: str,
        error_message: str,
    ) -> EvidenceReviewRunRecord:
        timestamp = self._now()
        progress = {
            "stage": "failed",
            "message": "人工取证任务失败",
        }
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_review_runs
                SET status = 'failed', progress_json = ?, completed_at = ?,
                    error_message = ?
                WHERE run_id = ? AND status IN ('queued', 'processing')
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    timestamp,
                    error_message,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise EvidenceRunTransitionError(run_id)
        return self.get_evidence_run(run_id)  # type: ignore[return-value]

    def recover_incomplete_evidence_runs(self, reason: str) -> int:
        timestamp = self._now()
        progress = {
            "stage": "failed",
            "message": "人工取证任务已中断",
        }
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_review_runs
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
