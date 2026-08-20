from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from mineru_raw_parse import DEFAULT_BACKEND, DEFAULT_SVR_URL


load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    data_dir: Path
    database_path: Path
    contracts_dir: Path
    mineru_url: str
    mineru_backend: str
    mineru_server_url: str | None
    image_vision_model: str | None = None
    image_vision_timeout_seconds: float = 120.0


def load_settings(base_dir: Path | None = None) -> Settings:
    project_dir = base_dir or Path(__file__).resolve().parent.parent
    data_dir = Path(os.getenv("APP_DATA_DIR", str(project_dir / "data"))).expanduser()
    database_path = Path(
        os.getenv("APP_DATABASE_PATH", str(data_dir / "contracts.db"))
    ).expanduser()
    contracts_dir = Path(
        os.getenv("APP_CONTRACTS_DIR", str(data_dir / "contracts"))
    ).expanduser()
    return Settings(
        project_dir=project_dir,
        data_dir=data_dir,
        database_path=database_path,
        contracts_dir=contracts_dir,
        mineru_url=os.getenv("PDF_TRANS_MINERU_URL", DEFAULT_SVR_URL),
        mineru_backend=os.getenv("PDF_TRANS_MINERU_BACKEND", DEFAULT_BACKEND),
        mineru_server_url=os.getenv("PDF_TRANS_MINERU_SERVER_URL") or None,
        image_vision_model=os.getenv("IMAGE_VISION_MODEL")
        or os.environ["LLM_MODEL"],
        image_vision_timeout_seconds=float(
            os.getenv("IMAGE_VISION_TIMEOUT_SECONDS", "120")
        ),
    )
