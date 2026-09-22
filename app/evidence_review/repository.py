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


class RuleSetTransitionError(RuntimeError):
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
