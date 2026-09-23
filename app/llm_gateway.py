from __future__ import annotations

from threading import BoundedSemaphore
from typing import Any

MAX_CONCURRENT_LLM_CALLS = 5
_slots = BoundedSemaphore(MAX_CONCURRENT_LLM_CALLS)


def invoke_llm(model: Any, prompt: Any) -> Any:
    """Limit simultaneous chat-model requests within this service process."""
    with _slots:
        return model.invoke(prompt)
