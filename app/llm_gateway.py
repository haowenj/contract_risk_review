from __future__ import annotations

import os
from threading import BoundedSemaphore
from typing import Any

MAX_CONCURRENT_LLM_CALLS = 5
_slots = BoundedSemaphore(MAX_CONCURRENT_LLM_CALLS)


def llm_backend_kwargs() -> dict[str, Any]:
    backend = os.getenv("LLM_BACKEND", "bailian").strip().lower()
    if backend == "bailian":
        return {}
    if backend == "vllm":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def invoke_llm(model: Any, prompt: Any) -> Any:
    """Limit simultaneous chat-model requests within this service process."""
    with _slots:
        return model.invoke(prompt)
