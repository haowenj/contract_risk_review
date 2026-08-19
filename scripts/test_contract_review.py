from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.contract_review import build_default_contract_review_service


CONTRACT_ID = "b6d5e9c1-9ae1-45bf-91ed-fdfe0f7e772f"

REVIEW_RULE_TEXT = """
1. 项目付款期限不得超过验收后90日。
2. 乙方未经甲方书面同意不得分包或转包。
3. 合同应明确约定延期履约的违约责任。
""".strip()


EVENT_HEADINGS = {
    "review_items_parsed": "解析出的 ReviewItem",
    "review_item_started": "当前审查项",
    "evidence_retrieved": "RAG 命中的 Evidence",
    "empty_evidence_rerank_debug": "Evidence 为空，Rerank Top3 Debug",
    "retrieval_query_rewritten": "证据不足，改写检索问题",
    "review_item_completed": "本项风险结果",
    "review_summary": "最终汇总",
}


def print_progress(event: str, payload: dict[str, Any]) -> None:
    heading = EVENT_HEADINGS.get(event, event)
    print(f"\n=== {heading} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    service = build_default_contract_review_service(
        progress_callback=print_progress
    )
    service.run(CONTRACT_ID, REVIEW_RULE_TEXT)


if __name__ == "__main__":
    main()
