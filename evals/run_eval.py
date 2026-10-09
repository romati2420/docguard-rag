"""Evaluación contra el set de referencia, separando recuperación y calidad de respuesta.

Uso: python -m evals.run_eval   (requiere GOOGLE_API_KEY e índice construido)
"""
import json
import time
from pathlib import Path

from docguard.app import get_agent
from docguard.citations import validate_answer
from docguard.graph import ask

GOLDEN = json.loads((Path(__file__).parent / "golden_set.json").read_text(encoding="utf-8"))
REPORT = Path(__file__).parent / "report.json"


def keyword_recall(text: str, keywords: list[str]) -> float:
    text = text.lower()
    return sum(k in text for k in keywords) / len(keywords) if keywords else 1.0


def run():
    graph, retriever = get_agent()
    rows = []
    for i, case in enumerate(GOLDEN):
        thread = f"eval-{case['id']}-{int(time.time())}"
        # 1) Recuperación, evaluada sin el LLM: ¿los fragmentos traen la información esperada?
        docs = retriever(case["question"], case["role"])
        context = " ".join(d.page_content for d in docs)
        retrieval_recall = keyword_recall(context, case.get("expected_keywords", []))

        # 2) Respuesta de punta a punta.
        t0 = time.perf_counter()
        answer = ask(graph, case["question"], case["role"], thread)
        latency = time.perf_counter() - t0
        text = answer.answer.lower()
        leaked = any(k in text for k in case.get("forbidden_keywords", []))
        row = {
            "id": case["id"],
            "retrieval_recall": round(retrieval_recall, 2) if case.get("expected_keywords") else None,
            "answer_recall": round(keyword_recall(text, case.get("expected_keywords", [])), 2)
            if case.get("expected_keywords") else None,
            "answerable_ok": answer.answerable == case["answerable"],
            "citations_valid": answer.answerable and not validate_answer(answer, docs) if case["answerable"] else None,
            "leak": leaked,
            "confidence": answer.confidence,
            "latency_s": round(latency, 2),
        }
        rows.append(row)
        print(f"[{i + 1}/{len(GOLDEN)}] {row}")

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "casos": len(rows),
        "retrieval_recall_promedio": mean("retrieval_recall"),
        "answer_recall_promedio": mean("answer_recall"),
        "decision_responder_correcta": mean("answerable_ok"),
        "citas_validas": mean("citations_valid"),
        "fugas": sum(r["leak"] for r in rows),
        "latencia_promedio_s": mean("latency_s"),
    }
    REPORT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResumen:", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
