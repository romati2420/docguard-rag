"""Monitoreo posterior al despliegue: agrega las trazas JSONL en indicadores operativos."""
import json
from collections import defaultdict

from . import config


def summarize() -> dict:
    if not config.TRACE_FILE.exists():
        return {"consultas": 0}
    events = [json.loads(line) for line in config.TRACE_FILE.read_text(encoding="utf-8").splitlines() if line]
    by_node = defaultdict(list)
    for e in events:
        by_node[e["node"]].append(e)

    def count(node, key=None, value=True):
        return sum(1 for e in by_node[node] if key is None or e.get(key) == value)

    finals = by_node["finalize"]
    tokens_in = sum((e.get("usage") or {}).get("input_tokens", 0) for e in by_node["generate"])
    tokens_out = sum((e.get("usage") or {}).get("output_tokens", 0) for e in by_node["generate"])
    total_queries = len(by_node["guard_input"])
    return {
        "consultas": total_queries,
        "bloqueadas_por_injection": count("guard_input", "blocked"),
        "sin_contexto_autorizado": count("no_context"),
        "abstenciones": sum(1 for e in finals if not e.get("answerable")),
        "respuestas_confianza_baja": sum(1 for e in finals if e.get("confidence") == "baja"),
        "reintentos_por_citas": sum(1 for e in by_node["generate"] if e.get("attempt", 1) > 1),
        "lineas_en_cuarentena": sum(e.get("quarantined_lines", 0) for e in by_node["retrieve"]),
        "acciones_denegadas_por_permisos": count("authorize_action", "allowed", False),
        "acciones_aprobadas": count("human_review", "approved", True),
        "acciones_rechazadas": count("human_review", "approved", False),
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "costo_estimado_usd": round(sum(e.get("cost_usd", 0) for e in by_node["generate"]), 4),
        "latencia_promedio_ms": {n: round(sum(e["latency_ms"] for e in ev) / len(ev), 1)
                                 for n, ev in by_node.items() if ev},
    }
