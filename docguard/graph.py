"""Grafo del agente (LangGraph): guardrail → recuperación con ACL → generación → verificación de citas."""
import operator
from typing import Annotated, Callable, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import config
from .citations import keep_valid_citations, validate_answer
from .guardrails import detect_injection, redact_pii, strip_injected_lines
from .schemas import Answer
from .tracing import estimate_cost, span

HISTORY_TURNS = 3


class AgentState(TypedDict, total=False):
    question: str
    role: str
    safe_question: str
    blocked_reason: str | None
    docs: list[Document]
    quarantined: int
    answer: Answer | None
    attempts: int
    citation_errors: list[str]
    history: Annotated[list[dict], operator.add]  # memoria conversacional por thread_id


def build_graph(retriever: Callable[[str, str], list[Document]], answer_fn: Callable, checkpointer=None):
    def guard_input(state: AgentState):
        with span("guard_input") as extra:
            q = state["question"]
            pattern = detect_injection(q)
            safe_q, pii = redact_pii(q)
            extra.update(blocked=bool(pattern), pii_found=pii)
            # Reinicia los campos por turno: el checkpointer conserva el estado entre turnos.
            return {"safe_question": safe_q, "blocked_reason": pattern, "docs": [], "quarantined": 0,
                    "answer": None, "attempts": 0, "citation_errors": []}

    def retrieve(state: AgentState):
        with span("retrieve", role=state["role"]) as extra:
            docs = retriever(state["safe_question"], state["role"])
            # Prompt injection indirecta: se eliminan las líneas con instrucciones incrustadas y,
            # si el fragmento queda vacío, se descarta completo.
            clean, removed = [], 0
            for d in docs:
                text, n = strip_injected_lines(d.page_content)
                removed += n
                if text:
                    clean.append(Document(text, metadata=d.metadata))
            extra.update(retrieved=len(docs), quarantined_lines=removed,
                         sources=[f"{d.metadata['source']}#p{d.metadata['page']}" for d in clean])
            return {"docs": clean, "quarantined": removed}

    def generate(state: AgentState):
        with span("generate", attempt=state["attempts"] + 1) as extra:
            history = state.get("history", [])[-HISTORY_TURNS:]
            answer, usage = answer_fn(state["safe_question"], state["docs"], history, state["citation_errors"])
            extra.update(usage=usage, cost_usd=round(estimate_cost(usage), 6))
            return {"answer": answer, "attempts": state["attempts"] + 1}

    def validate(state: AgentState):
        with span("validate") as extra:
            errors = validate_answer(state["answer"], state["docs"])
            extra.update(errors=len(errors))
            return {"citation_errors": errors}

    def finalize(state: AgentState):
        with span("finalize") as extra:
            answer = state["answer"].model_copy()
            if state["citation_errors"]:
                # Calibración: lo que no se puede respaldar con fuentes no se entrega como seguro.
                answer.citations = keep_valid_citations(answer, state["docs"])
                answer.confidence = "baja"
                if not answer.citations:
                    answer = Answer(answer="No pude respaldar una respuesta con citas verificables en los documentos.",
                                    answerable=False, confidence="baja")
            answer.answer, pii = redact_pii(answer.answer)
            extra.update(answerable=answer.answerable, confidence=answer.confidence, pii_redacted=pii)
            return {"answer": answer, "history": [{"question": state["safe_question"], "answer": answer.answer}]}

    def refuse(state: AgentState):
        with span("refuse", pattern=state["blocked_reason"]):
            answer = Answer(answer="Solicitud bloqueada por la política de seguridad: parece un intento de "
                                   "modificar las instrucciones del asistente.",
                            answerable=False, confidence="alta")
            return {"answer": answer, "history": [{"question": "[bloqueada]", "answer": answer.answer}]}

    def no_context(state: AgentState):
        with span("no_context"):
            answer = Answer(answer="No encontré información sobre esto en los documentos autorizados para tu rol.",
                            answerable=False, confidence="alta")
            return {"answer": answer, "history": [{"question": state["safe_question"], "answer": answer.answer}]}

    def after_guard(state: AgentState):
        return "refuse" if state["blocked_reason"] else "retrieve"

    def after_retrieve(state: AgentState):
        return "generate" if state["docs"] else "no_context"

    def after_validate(state: AgentState):
        if state["citation_errors"] and state["attempts"] < config.MAX_GENERATION_ATTEMPTS:
            return "generate"  # reintento con retroalimentación sobre las citas inválidas
        return "finalize"

    g = StateGraph(AgentState)
    for name, fn in [("guard_input", guard_input), ("retrieve", retrieve), ("generate", generate),
                     ("validate", validate), ("finalize", finalize), ("refuse", refuse), ("no_context", no_context)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["refuse", "retrieve"])
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "no_context"])
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, ["generate", "finalize"])
    for terminal in ("finalize", "refuse", "no_context"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def ask(graph, question: str, role: str, thread_id: str = "default") -> Answer:
    result = graph.invoke({"question": question, "role": role}, {"configurable": {"thread_id": thread_id}})
    return result["answer"]
