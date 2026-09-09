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

    def test_cache_does_not_reuse_an_index_from_another_contract_version(self):
        with TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            manager = IndexManager(embedding_model=object())
            first = object()
            second = object()
            manager.put("c1", first, index_version="v1")
            contract = SimpleNamespace(
                contract_id="c1",
                index_version="v2",
                storage_dir=temp_dir,
            )

            with patch("app.index_manager.StorageContext.from_defaults"), patch(
                "app.index_manager.load_index_from_storage",
                return_value=second,
            ):
                loaded = manager.get(contract)

        self.assertIs(loaded, second)

    def test_bm25_cache_miss_loads_persisted_index_with_contract_version(self):
        with TemporaryDirectory() as temp_dir:
            bm25_dir = Path(temp_dir) / "bm25_index"
            bm25_dir.mkdir()
            manager = IndexManager(embedding_model=object())
            contract = SimpleNamespace(
                contract_id="c1",
                index_version="v2",
                storage_dir=temp_dir,
            )
            loaded_index = object()

            with patch(
                "app.index_manager.BM25Index.load", return_value=loaded_index
            ) as loader:
                self.assertIs(manager.get_bm25(contract), loaded_index)

        loader.assert_called_once_with(bm25_dir)

    def test_bm25_cache_is_cleared_with_vector_cache(self):
        manager = IndexManager(embedding_model=object())
        bm25_index = object()
        manager.put_bm25("c1", bm25_index, index_version="v1")

        manager.clear("c1")

        with patch("app.index_manager.BM25Index.load") as loader:
            contract = SimpleNamespace(
                contract_id="c1",
                index_version="v1",
                storage_dir="/not-used",
            )
            try:
                manager.get_bm25(contract)
            except FileNotFoundError:
                pass
        loader.assert_not_called()
