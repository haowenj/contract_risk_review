from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from app.index_manager import IndexManager


class IndexManagerTest(TestCase):
    def test_cache_hit_does_not_load_storage(self):
        manager = IndexManager(embedding_model=object())
        index = object()
        manager.put("c1", index)
        contract = SimpleNamespace(contract_id="c1", storage_dir="/not-used")

        with patch("app.index_manager.load_index_from_storage") as loader:
            self.assertIs(manager.get(contract), index)

        loader.assert_not_called()

    def test_cache_miss_loads_persisted_index_with_embedding_model(self):
        with TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            embedding_model = object()
            loaded_index = object()
            manager = IndexManager(embedding_model=embedding_model)
            contract = SimpleNamespace(contract_id="c1", storage_dir=temp_dir)

            with patch("app.index_manager.StorageContext.from_defaults") as storage, patch(
                "app.index_manager.load_index_from_storage", return_value=loaded_index
            ) as loader:
                self.assertIs(manager.get(contract), loaded_index)

        storage.assert_called_once_with(persist_dir=str(index_dir))
        loader.assert_called_once_with(storage.return_value, embed_model=embedding_model)
