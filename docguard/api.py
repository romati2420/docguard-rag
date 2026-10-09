"""API HTTP (FastAPI) + interfaz web, con autenticación por token.

- El rol y la identidad salen del token (servidor), nunca del cuerpo de la petición.
- Las conversaciones se aíslan por usuario: el thread_id interno es "<usuario>:<thread_id>".
- Solo un revisor puede aprobar acciones, y nunca las que él mismo solicitó.
"""
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import metrics
from .app import get_agent
from .auth import User, current_user
from .graph import review, run

app = FastAPI(title="DocGuard", description="Agente RAG documental seguro (LangGraph + Gemini)")
STATIC = Path(__file__).parent / "static"

# Acciones esperando revisión humana: review_id → datos de la solicitud. En memoria (prototipo).
PENDING: dict[str, dict] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(default="default", max_length=64, pattern=r"^[\w-]+$")


class ReviewRequest(BaseModel):
    review_id: str
    approved: bool


def _thread(user: User, thread_id: str) -> str:
    return f"{user.name}:{thread_id}"


def _register_pending(review_id: str, pending: dict) -> dict:
    PENDING[review_id] = {"review_id": review_id, **pending}
    return PENDING[review_id]


def _serialize(result: dict) -> dict:
    return {**result, "answer": result["answer"].model_dump() if result["answer"] else None}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def me(user: User = Depends(current_user)):
    return {"user": user.name, "role": user.role, "reviewer": user.reviewer}


@app.get("/metrics")
def get_metrics(user: User = Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo admin puede ver las métricas")
    return metrics.summarize()


@app.post("/ask")
def ask_endpoint(req: AskRequest, user: User = Depends(current_user)):
    graph, _ = get_agent()
    thread = _thread(user, req.thread_id)
    result = run(graph, req.question, user.role, thread, user=user.name)
    if result["pending_action"]:
        result["pending_action"] = _register_pending(thread, result["pending_action"])
    return _serialize(result)


@app.get("/reviews/pending")
def pending_reviews(user: User = Depends(current_user)):
    if not user.reviewer:
        raise HTTPException(status_code=403, detail="El usuario no tiene permiso de revisión")
    return [p for p in PENDING.values() if p["requested_by"] != user.name]


@app.post("/review")
def review_endpoint(req: ReviewRequest, user: User = Depends(current_user)):
    if not user.reviewer:
        raise HTTPException(status_code=403, detail="El usuario no tiene permiso de revisión")
    pending = PENDING.get(req.review_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="No hay una acción pendiente con ese ID")
    if pending["requested_by"] == user.name:
        raise HTTPException(status_code=403, detail="Separación de funciones: no puedes aprobar tu propia solicitud")
    graph, _ = get_agent()
    result = review(graph, req.review_id, req.approved, reviewer=user.name)
    PENDING.pop(req.review_id, None)
    return _serialize(result)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, user: User = Depends(current_user)):
    graph, _ = get_agent()
    thread = _thread(user, req.thread_id)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def events():
        config = {"configurable": {"thread_id": thread}}
        state = {"question": req.question, "role": user.role, "requested_by": user.name}
        for update in graph.stream(state, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    yield sse("pending_action", _register_pending(thread, data[0].value))
                    continue
                yield sse("node", {"node": node})
                if data and data.get("answer") is not None and node in {"finalize", "refuse", "no_context"}:
                    yield sse("answer", data["answer"].model_dump())
                if data and data.get("action_status") == "denegada_por_permisos":
                    yield sse("action_denied", {"tool": "crear_ticket", "role": user.role})
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
