from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractRecord:
    contract_id: str
    filename: str
    storage_dir: str
    index_version: str | None
    status: str
    error_message: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "contract_id": self.contract_id,
            "filename": self.filename,
            "storage_dir": self.storage_dir,
            "index_version": self.index_version,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
