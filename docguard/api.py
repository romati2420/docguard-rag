"""API HTTP (FastAPI) + interfaz web: preguntas, streaming SSE, aprobación humana y métricas."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import metrics
from .app import get_agent
from .graph import review, run

app = FastAPI(title="DocGuard", description="Agente RAG documental seguro (LangGraph + Gemini)")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # En producción el rol vendría del token de identidad del usuario, no del cuerpo de la petición.
    role: str = "analista"
    thread_id: str = "default"


class ReviewRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer: str = Field(default="revisor", max_length=80)


def _serialize(result: dict) -> dict:
    return {**result, "answer": result["answer"].model_dump() if result["answer"] else None}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return metrics.summarize()


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    graph, _ = get_agent()
    return _serialize(run(graph, req.question, req.role, req.thread_id))


@app.post("/review")
def review_endpoint(req: ReviewRequest):
    graph, _ = get_agent()
    return _serialize(review(graph, req.thread_id, req.approved, req.reviewer))


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    graph, _ = get_agent()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def events():
        config = {"configurable": {"thread_id": req.thread_id}}
        for update in graph.stream({"question": req.question, "role": req.role}, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    yield sse("pending_action", data[0].value)
                    continue
                yield sse("node", {"node": node})
                if data and data.get("answer") is not None and node in {"finalize", "refuse", "no_context"}:
                    yield sse("answer", data["answer"].model_dump())
                if data and data.get("action_status") == "denegada_por_permisos":
                    yield sse("action_denied", {"tool": "crear_ticket", "role": req.role})
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
