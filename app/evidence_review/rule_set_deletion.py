from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    RuleSetDeletionNotAllowedError,
)


def delete_rule_set(
    repository: EvidenceReviewRepository,
    rule_sets_dir: Path,
    rule_set_id: str,
) -> None:
    record = repository.get_rule_set(rule_set_id)
    if record is None:
        raise KeyError(rule_set_id)
    if record.status in {"queued", "processing"}:
        raise RuleSetDeletionNotAllowedError("规则文件仍在解析，完成后才能删除。")

    root = Path(rule_sets_dir).expanduser().resolve()
    upload_root = root / "uploads"
    source_path = Path(record.source_path).expanduser().resolve()
    upload_dir = source_path.parent
    parsed_dir = (root / rule_set_id).resolve()
    try:
        uuid.UUID(upload_dir.name)
    except ValueError as exc:
        raise RuleSetDeletionNotAllowedError(
            "规则文件存储位置不符合预期，无法安全删除。"
        ) from exc
    if (
        upload_dir.parent != upload_root
        or source_path.name not in {"source.pdf", "source.txt", "source.md"}
        or parsed_dir.parent != root
        or parsed_dir.name != rule_set_id
        or any(path.exists() and not path.is_dir() for path in (upload_dir, parsed_dir))
    ):
        raise RuleSetDeletionNotAllowedError(
            "规则文件存储位置不符合预期，无法安全删除。"
        )

    staged: list[tuple[Path, Path]] = []
    try:
        for directory in (upload_dir, parsed_dir):
            if directory.exists():
                renamed = directory.parent / f".deleting-{directory.name}-{uuid.uuid4().hex}"
                directory.rename(renamed)
                staged.append((directory, renamed))
        repository.delete_finished_rule_set(rule_set_id)
    except Exception:
        for original, renamed in reversed(staged):
            renamed.rename(original)
        raise

    for _, renamed in staged:
        shutil.rmtree(renamed)
