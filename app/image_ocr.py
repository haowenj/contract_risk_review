from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mineru_raw_parse import run_image_ocr


class OCRRunner(Protocol):
    def __call__(
        self,
        image_path: Path,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
    ) -> str:
        ...


class MinerUImageOCRService:
    def __init__(
        self,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
        runner: OCRRunner = run_image_ocr,
    ) -> None:
        self._svr_url = svr_url
        self._backend = backend
        self._server_url = server_url
        self._runner = runner

    def extract_text(self, image_path: Path) -> str:
        return self._runner(
            image_path,
            svr_url=self._svr_url,
            backend=self._backend,
            server_url=self._server_url,
        )
