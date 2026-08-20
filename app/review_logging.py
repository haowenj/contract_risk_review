from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

logger = logging.getLogger(__name__)
MAX_DIAGNOSTIC_LIST_ITEMS = 20
MAX_EVIDENCE_TEXT_CHARS = 2_000
MAX_DIAGNOSTIC_TEXT_CHARS = 8_000
MAX_ITEM_DIAGNOSTIC_BYTES = 256 * 1024


def _bounded_diagnostic_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _bounded_diagnostic_value(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        retained = [
            _bounded_diagnostic_value(child)
            for child in value[:MAX_DIAGNOSTIC_LIST_ITEMS]
        ]
        omitted = len(value) - len(retained)
        if omitted:
            retained.append({"_truncated": True, "_omitted_count": omitted})
        return retained
    if isinstance(value, str):
        limit = (
            MAX_EVIDENCE_TEXT_CHARS
            if key in {"evidence_text", "text"}
            else MAX_DIAGNOSTIC_TEXT_CHARS
        )
        return value[:limit]
    return value


class ReviewRunJournal:
    """Best-effort, file-only diagnostics for one contract review run."""

    def __init__(self, root_dir: Path, run_id: str):
        self.run_dir = Path(root_dir) / run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.current_item_path = self.run_dir / "current_item.jsonl.tmp"
        self._item: dict[str, Any] | None = None
        self._item_started_at = 0.0
        self._retrieval_attempts = 0
        self._used_absence_check = False
        self._item_detail_truncated = False
        self._first_retrieval_success = 0
        self._second_retrieval_success = 0
        self._insufficient = 0
        self._absence_verified = 0

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _json_line(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str) + "\n"

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(self._json_line(payload))
        except Exception:
            logger.exception("unable to persist contract review diagnostics")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except Exception:
            logger.exception("unable to persist contract review diagnostics")

    def _append_current_item(self, payload: dict[str, Any]) -> None:
        if self._item_detail_truncated:
            return
        try:
            existing_size = (
                self.current_item_path.stat().st_size
                if self.current_item_path.exists()
                else 0
            )
            next_size = len(self._json_line(payload).encode("utf-8"))
            if existing_size + next_size > MAX_ITEM_DIAGNOSTIC_BYTES:
                self._append(
                    self.current_item_path,
                    {
                        "timestamp": self._timestamp(),
                        "event": "diagnostic_truncated",
                        "max_bytes": MAX_ITEM_DIAGNOSTIC_BYTES,
                    },
                )
                self._item_detail_truncated = True
                return
            self._append(self.current_item_path, payload)
        except Exception:
            logger.exception("unable to bound contract review item diagnostics")

    def started(self, contract_id: str) -> None:
        self._append(
            self.events_path,
            {
                "timestamp": self._timestamp(),
                "event": "run_started",
                "contract_id": contract_id,
            },
        )

    def created(
        self,
        *,
        contract_id: str,
        review_rule_text: str,
        created_at: str,
    ) -> None:
        self._write_json(
            self.run_dir / "input.json",
            {
                "run_id": self.run_dir.name,
                "contract_id": contract_id,
                "review_rule_text": review_rule_text,
                "created_at": created_at,
            },
        )

    def record(self, event: str, payload: dict[str, Any]) -> None:
        if event == "review_items_parsed":
            review_items = payload.get("review_items")
            total = len(review_items) if isinstance(review_items, list) else 0
            self._append(
                self.events_path,
                {
                    "timestamp": self._timestamp(),
                    "event": "rules_parsed",
                    "total_items": total,
                },
            )
            return

        if event == "review_item_started":
            self._item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
            self._item_started_at = monotonic()
            self._retrieval_attempts = 0
            self._used_absence_check = False
            self._item_detail_truncated = False
            try:
                self.current_item_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("unable to reset contract review item diagnostics")

        if self._item is not None:
            if event == "evidence_retrieved":
                try:
                    self._retrieval_attempts = max(
                        self._retrieval_attempts,
                        int(payload.get("attempt", 0)),
                    )
                except (TypeError, ValueError):
                    pass
            elif event == "retrieval_query_rewritten":
                self._retrieval_attempts = max(self._retrieval_attempts, 2)
            elif event == "absence_check_started":
                self._used_absence_check = True

            self._append_current_item(
                {
                    "timestamp": self._timestamp(),
                    "event": event,
                    "payload": _bounded_diagnostic_value(payload),
                },
            )

        if event == "review_item_completed":
            self._complete_item(payload)

    def _complete_item(self, payload: dict[str, Any]) -> None:
        result = payload.get("result")
        result = result if isinstance(result, dict) else {}
        evidence = result.get("evidence")
        evidence = evidence if isinstance(evidence, list) else []
        evidence_locations = [
            {
                "page_idx": value.get("page_idx"),
                "source_object_index": value.get("source_object_index"),
                "node_type": value.get("node_type"),
            }
            for value in evidence
            if isinstance(value, dict)
        ]
        evidence_status = result.get("evidence_status")
        if evidence_status == "found" and not self._used_absence_check:
            if self._retrieval_attempts == 2:
                self._second_retrieval_success += 1
            else:
                self._first_retrieval_success += 1
        elif evidence_status == "insufficient":
            self._insufficient += 1
        elif evidence_status == "absence_verified":
            self._absence_verified += 1
        compact = (
            evidence_status == "found"
            and result.get("risk_status") != "needs_review"
            and not self._used_absence_check
        )

        if compact:
            self._append(
                self.events_path,
                {
                    "timestamp": self._timestamp(),
                    "event": "item_completed",
                    "detail": "summary",
                    "item_id": result.get("item_id") or (self._item or {}).get("id"),
                    "item_name": result.get("item_name") or (self._item or {}).get("name"),
                    "risk_status": result.get("risk_status"),
                    "risk_level": result.get("risk_level"),
                    "evidence_status": evidence_status,
                    "retrieval_attempts": self._retrieval_attempts or 1,
                    "recovered_by_second_retrieval": self._retrieval_attempts == 2,
                    "evidence_count": len(evidence),
                    "evidence_locations": evidence_locations,
                    "elapsed_ms": round((monotonic() - self._item_started_at) * 1000),
                },
            )
            try:
                self.current_item_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("unable to remove compacted review diagnostics")
        else:
            self._flush_current_item()

        self._item = None

    def _flush_current_item(self) -> None:
        try:
            if self.current_item_path.is_file():
                self.events_path.parent.mkdir(parents=True, exist_ok=True)
                with self.events_path.open("a", encoding="utf-8") as target:
                    target.write(self.current_item_path.read_text(encoding="utf-8"))
                self.current_item_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("unable to flush contract review item diagnostics")

    def completed(self, result: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "result.json", result)
        self._append(
            self.events_path,
            {
                "timestamp": self._timestamp(),
                "event": "diagnostic_summary",
                "first_retrieval_success": self._first_retrieval_success,
                "second_retrieval_success": self._second_retrieval_success,
                "insufficient": self._insufficient,
                "absence_verified": self._absence_verified,
            },
        )
        self._append(
            self.events_path,
            {"timestamp": self._timestamp(), "event": "run_completed"},
        )

    def failed(self, exc: Exception) -> None:
        self._flush_current_item()
        self._write_json(
            self.run_dir / "failure.json",
            {
                "timestamp": self._timestamp(),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )
        self._append(
            self.events_path,
            {
                "timestamp": self._timestamp(),
                "event": "run_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
