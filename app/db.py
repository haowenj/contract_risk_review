from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator

from app.models import ContractRecord


CONTRACT_STATUSES = frozenset({"queued", "processing", "ready", "failed"})


class ContractRepository:
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS contracts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    storage_dir TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'ready', 'failed')
                    ),
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _record_from_row(row: sqlite3.Row | None) -> ContractRecord | None:
        if row is None:
            return None
        return ContractRecord(
            contract_id=row["contract_id"],
            filename=row["filename"],
            storage_dir=row["storage_dir"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self,
        filename: str,
        storage_dir: Path,
        contract_id: str | None = None,
    ) -> ContractRecord:
        contract_id = contract_id or str(uuid.uuid4())
        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO contracts (
                    contract_id, filename, storage_dir, status,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', NULL, ?, ?)
                """,
                (contract_id, filename, str(storage_dir), timestamp, timestamp),
            )
        return self.get(contract_id)  # type: ignore[return-value]

    def get(self, contract_id: str) -> ContractRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM contracts WHERE contract_id = ?",
                (contract_id,),
            ).fetchone()
        return self._record_from_row(row)

    def list(self) -> list[ContractRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM contracts ORDER BY sequence DESC"
            ).fetchall()
        return [record for row in rows if (record := self._record_from_row(row))]

    def update_status(
        self,
        contract_id: str,
        status: str,
        error_message: str | None = None,
    ) -> ContractRecord:
        if status not in CONTRACT_STATUSES:
            raise ValueError(f"unsupported contract status: {status}")

        timestamp = self._now()
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE contracts
                SET status = ?, error_message = ?, updated_at = ?
                WHERE contract_id = ?
                """,
                (status, error_message, timestamp, contract_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(contract_id)
        return self.get(contract_id)  # type: ignore[return-value]
