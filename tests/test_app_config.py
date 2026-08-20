from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.config import load_settings


class SettingsTest(TestCase):
    def test_missing_llm_model_does_not_fall_back_to_provider_default(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(KeyError, "LLM_MODEL"),
        ):
            load_settings(Path("/project"))

    def test_defaults_are_relative_to_requested_project_directory(self):
        with TemporaryDirectory() as temp_dir:
            settings = load_settings(Path(temp_dir))

        self.assertEqual(settings.data_dir, Path(temp_dir) / "data")
        self.assertEqual(settings.database_path, Path(temp_dir) / "data" / "contracts.db")
        self.assertEqual(settings.contracts_dir, Path(temp_dir) / "data" / "contracts")

    def test_environment_variables_override_storage_paths(self):
        with patch.dict(
            "os.environ",
            {
                "APP_DATA_DIR": "/tmp/rag-data",
                "APP_DATABASE_PATH": "/tmp/rag.db",
                "APP_CONTRACTS_DIR": "/tmp/contracts",
            },
            clear=False,
        ):
            settings = load_settings(Path("/project"))

        self.assertEqual(settings.data_dir, Path("/tmp/rag-data"))
        self.assertEqual(settings.database_path, Path("/tmp/rag.db"))
        self.assertEqual(settings.contracts_dir, Path("/tmp/contracts"))

    def test_load_settings_reads_image_vision_configuration(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL": "qwen-current-plus",
                "IMAGE_VISION_MODEL": "qwen-test-vl",
                "IMAGE_VISION_TIMEOUT_SECONDS": "180",
            },
            clear=False,
        ):
            settings = load_settings(Path("/project"))

        self.assertEqual(settings.image_vision_model, "qwen-test-vl")
        self.assertEqual(settings.image_vision_timeout_seconds, 180.0)
