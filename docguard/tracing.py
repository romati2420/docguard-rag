"""Instrumentación mínima: latencia, tokens y costo estimado por nodo, en JSONL."""
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config


def estimate_cost(usage: dict | None) -> float:
    if not usage:
        return 0.0
    return (usage.get("input_tokens", 0) * config.PRICE_INPUT_PER_M
            + usage.get("output_tokens", 0) * config.PRICE_OUTPUT_PER_M) / 1_000_000


def log_event(event: dict) -> None:
    config.TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with config.TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@contextmanager
def span(node: str, **attrs):
    """Mide un nodo del grafo; el bloque puede agregar datos a `extra` (tokens, conteos)."""
    extra: dict = {}
    start = time.perf_counter()
    try:
        yield extra
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event({"node": node, "latency_ms": latency_ms, **attrs, **extra})
