"""Thin LiteLLM wrapper that returns tokens and a reproducible cost."""
import logging
import time

import litellm

from .config import settings

log = logging.getLogger(__name__)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = settings()["prices"].get(model)
    if not price or (price["input"] == 0 and price["output"] == 0):
        log.warning("No price set for %s in config.yaml; cost will read as 0", model)
        price = price or {"input": 0.0, "output": 0.0}
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def complete(model: str, messages: list[dict], **kwargs) -> dict:
    t0 = time.perf_counter()
    resp = litellm.completion(model=model, messages=messages, temperature=0, **kwargs)
    usage = resp.usage
    return {
        "text": resp.choices[0].message.content or "",
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost_usd(model, usage.prompt_tokens, usage.completion_tokens),
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
