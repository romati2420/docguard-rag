"""API HTTP (FastAPI): respuesta completa o streaming del progreso del agente por SSE."""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .app import get_agent
from .graph import ask
from .schemas import Answer

app = FastAPI(title="DocGuard", description="Agente RAG documental seguro (LangGraph + Gemini)")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # En producción el rol vendría del token de identidad del usuario, no del cuerpo de la petición.
    role: str = "analista"
    thread_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask_endpoint(req: AskRequest):
    graph, _ = get_agent()
    return ask(graph, req.question, req.role, req.thread_id)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    graph, _ = get_agent()

    def events():
        config = {"configurable": {"thread_id": req.thread_id}}
        for update in graph.stream({"question": req.question, "role": req.role}, config, stream_mode="updates"):
            for node, data in update.items():
                yield f"event: node\ndata: {json.dumps({'node': node})}\n\n"
                if data and data.get("answer") is not None and node in {"finalize", "refuse", "no_context"}:
                    yield f"event: answer\ndata: {data['answer'].model_dump_json()}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
