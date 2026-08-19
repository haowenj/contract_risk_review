import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.db import ContractRepository


class ContractRepositoryTest(TestCase):
    def test_read_closes_sqlite_connection(self):
        with TemporaryDirectory() as temp_dir:
            repository = ContractRepository(Path(temp_dir) / "contracts.db")
            connection = sqlite3.connect(repository.database_path)
            repository._connect = lambda: connection

            repository.get("missing")

            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_write_closes_sqlite_connection(self):
        with TemporaryDirectory() as temp_dir:
            repository = ContractRepository(Path(temp_dir) / "contracts.db")
            connection = sqlite3.connect(repository.database_path)
            repository._connect = lambda: connection

            with patch.object(repository, "get", return_value=None):
                repository.create("contract.pdf", Path(temp_dir) / "contract")

            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_create_update_get_and_list_contracts(self):
        with TemporaryDirectory() as temp_dir:
            repository = ContractRepository(Path(temp_dir) / "contracts.db")
            first = repository.create("a.pdf", Path(temp_dir) / "a")
            second = repository.create("b.pdf", Path(temp_dir) / "b")
            failed = repository.update_status(second.contract_id, "failed", "MinerU down")
            loaded = repository.get(second.contract_id)
            records = repository.list()

        self.assertEqual(first.status, "queued")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(loaded.error_message, "MinerU down")
        self.assertEqual([record.contract_id for record in records], [second.contract_id, first.contract_id])

    def test_update_status_rejects_unknown_contract(self):
        with TemporaryDirectory() as temp_dir:
            repository = ContractRepository(Path(temp_dir) / "contracts.db")

            with self.assertRaises(KeyError):
                repository.update_status("missing", "failed", "error")

    def test_ready_contract_persists_index_version(self):
        with TemporaryDirectory() as temp_dir:
            repository = ContractRepository(Path(temp_dir) / "contracts.db")
            contract = repository.create("contract.pdf", Path(temp_dir) / "contract")

            ready = repository.update_status(
                contract.contract_id,
                "ready",
                index_version="index-v2",
            )

            loaded = repository.get(contract.contract_id)

        self.assertEqual(ready.index_version, "index-v2")
        self.assertEqual(loaded.index_version, "index-v2")
        self.assertEqual(loaded.to_dict()["index_version"], "index-v2")

    def test_existing_contract_table_is_migrated_with_index_version(self):
        with TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "contracts.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                """
                CREATE TABLE contracts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    storage_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO contracts (
                    contract_id, filename, storage_dir, status,
                    error_message, created_at, updated_at
                ) VALUES ('legacy-id', 'legacy.pdf', '/tmp/legacy', 'ready', NULL, 'now', 'now')
                """
            )
            connection.commit()
            connection.close()

            repository = ContractRepository(database_path)
            columns = {
                row[1]
                for row in sqlite3.connect(database_path).execute(
                    "PRAGMA table_info(contracts)"
                )
            }
            loaded = repository.get("legacy-id")

        self.assertIn("index_version", columns)
        self.assertEqual(loaded.index_version, "legacy-legacy-id")
