from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping


VerificationStatus = Literal[
    "verified",
    "conflict",
    "insufficient",
    "not_required",
]


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    details: dict[str, Any]


_SEPARATOR_RE = re.compile(r"[\s\-—_/·.,，。:：()（）]+")
_ACCOUNT_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\d[\d\s\-—_/]{6,}\d)(?!\d)"
)
_IDENTITY_CANDIDATE_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_DATE_RE = re.compile(
    r"\d{4}\s*[年./-]\s*\d{1,2}\s*[月./-]\s*\d{1,2}\s*日?"
)


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).upper()
    return _SEPARATOR_RE.sub("", normalized)


def _normalize_digits(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value) if char.isdigit())


def _account_candidates(ocr_text: str) -> list[str]:
    candidates = {
        _normalize_digits(match)
        for match in _ACCOUNT_CANDIDATE_RE.findall(ocr_text)
    }
    return sorted(candidate for candidate in candidates if 8 <= len(candidate) <= 30)


def _identity_candidates(ocr_text: str) -> list[str]:
    return sorted(
        {
            _normalize(match)
            for match in _IDENTITY_CANDIDATE_RE.findall(ocr_text)
        }
    )


def _date_candidates(ocr_text: str) -> list[str]:
    return [_normalize_digits(match) for match in _DATE_RE.findall(ocr_text)]


def _label_candidates(ocr_text: str, labels: tuple[str, ...]) -> list[str]:
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:{label_pattern})\s*[：: ]\s*([^\n,，;；|]+)"
    )
    return [value.strip(" \t:：,，。") for value in pattern.findall(ocr_text)]


def _field_detail(
    vl_value: Any,
    ocr_text: str,
    candidates: list[str],
    *,
    normalize_candidate=_normalize,
) -> dict[str, Any]:
    normalized_vl = normalize_candidate(vl_value)
    normalized_candidates = [
        normalize_candidate(candidate)
        for candidate in candidates
        if normalize_candidate(candidate)
    ]
    unique_candidates = sorted(set(normalized_candidates))

    if not normalized_vl:
        status = "insufficient"
    elif normalized_vl in unique_candidates:
        status = "verified" if len(unique_candidates) == 1 else "conflict"
    elif unique_candidates:
        status = "conflict"
    elif normalized_vl in normalize_candidate(ocr_text):
        status = "verified"
    else:
        status = "insufficient"

    return {
        "status": status,
        "vl_value": vl_value,
        "ocr_candidates": candidates,
    }


def _date_detail(
    vl_value: Any,
    ocr_candidates: list[str],
    position: int,
) -> dict[str, Any]:
    normalized_vl = _normalize_digits(vl_value) if isinstance(vl_value, str) else ""
    if not normalized_vl:
        status = "insufficient"
    elif len(ocr_candidates) > position:
        status = "verified" if ocr_candidates[position] == normalized_vl else "conflict"
    elif normalized_vl in ocr_candidates:
        status = "verified"
    elif ocr_candidates:
        status = "conflict"
    else:
        status = "insufficient"
    return {
        "status": status,
        "vl_value": vl_value,
        "ocr_candidates": ocr_candidates,
    }


def _aggregate_status(
    details: dict[str, dict[str, Any]],
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> VerificationStatus:
    if any(detail.get("status") == "conflict" for detail in details.values()):
        return "conflict"
    if not all(details[field].get("status") == "verified" for field in required_fields):
        return "insufficient"
    for field in optional_fields:
        if field in details and details[field].get("vl_value") and details[field].get(
            "status"
        ) != "verified":
            return "insufficient"
    return "verified"


def verify_image_data(
    image_type: str,
    structured_data: Mapping[str, Any],
    ocr_text: str | None,
) -> VerificationResult:
    if image_type == "general":
        return VerificationResult(status="not_required", details={})

    text = ocr_text if isinstance(ocr_text, str) else ""
    if image_type == "bank_account":
        account_candidates = _account_candidates(text)
        details = {
            "account_name": _field_detail(
                structured_data.get("account_name"),
                text,
                _label_candidates(text, ("户名", "账户名称")),
            ),
            "account_number": _field_detail(
                structured_data.get("account_number"),
                text,
                account_candidates,
                normalize_candidate=_normalize_digits,
            ),
            "bank_name": _field_detail(
                structured_data.get("bank_name"),
                text,
                _label_candidates(text, ("开户银行", "开户行", "银行")),
            ),
        }
        if structured_data.get("bank_branch"):
            details["bank_branch"] = _field_detail(
                structured_data.get("bank_branch"),
                text,
                _label_candidates(text, ("开户支行", "支行")),
            )
        return VerificationResult(
            status=_aggregate_status(
                details,
                required_fields=("account_name", "account_number", "bank_name"),
                optional_fields=("bank_branch",),
            ),
            details=details,
        )

    if image_type == "identity_card":
        identity_candidates = _identity_candidates(text)
        dates = _date_candidates(text)
        details = {
            "name": _field_detail(
                structured_data.get("name"),
                text,
                _label_candidates(text, ("姓名",)),
            ),
            "id_number": _field_detail(
                structured_data.get("id_number"),
                text,
                identity_candidates,
                normalize_candidate=_normalize,
            ),
            "valid_from": _date_detail(structured_data.get("valid_from"), dates, 0),
            "valid_to": _date_detail(structured_data.get("valid_to"), dates, 1),
        }
        return VerificationResult(
            status=_aggregate_status(
                details,
                required_fields=("name", "id_number"),
                optional_fields=("valid_from", "valid_to"),
            ),
            details=details,
        )

    return VerificationResult(status="insufficient", details={})
