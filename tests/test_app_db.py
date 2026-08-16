from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.db import ContractRepository


class ContractRepositoryTest(TestCase):
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
