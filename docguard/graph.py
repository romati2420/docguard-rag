"""Grafo del agente (LangGraph): guardrail → recuperación con ACL → generación → verificación de citas
→ (si se propone una acción) permisos de herramienta → aprobación humana → ejecución."""
import operator
from typing import Annotated, Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import config
from .citations import keep_valid_citations, validate_answer
from .guardrails import detect_injection, redact_pii, strip_injected_lines
from .schemas import Answer
from .tools import can_use_tool, create_ticket
from .tracing import estimate_cost, span

HISTORY_TURNS = 3


class AgentState(TypedDict, total=False):
    question: str
    role: str
    requested_by: str  # usuario autenticado que hace la consulta
    safe_question: str
    blocked_reason: str | None
    docs: list[Document]
    quarantined: int
    answer: Answer | None
    attempts: int
    citation_errors: list[str]
    action_status: str | None  # pendiente_aprobacion | denegada_por_permisos | aprobada | rechazada | ejecutada
    ticket: dict | None
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
                    "answer": None, "attempts": 0, "citation_errors": [], "action_status": None, "ticket": None}

    def retrieve(state: AgentState):
        with span("retrieve", role=state["role"]) as extra:
            docs = retriever(state["safe_question"], state["role"])
            # Prompt injection indirecta: se eliminan las líneas con instrucciones incrustadas y,
            # si el fragmento queda vacío, se descarta completo.
            # Además se redacta la PII de los fragmentos antes de que lleguen al modelo.
            clean, removed, pii_in_docs = [], 0, 0
            for d in docs:
                text, n = strip_injected_lines(d.page_content)
                text, pii = redact_pii(text)
                removed += n
                pii_in_docs += len(pii)
                if text:
                    clean.append(Document(text, metadata=d.metadata))
            extra.update(retrieved=len(docs), quarantined_lines=removed, pii_redacted_in_docs=pii_in_docs,
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
            action = answer.proposed_action
            if state["citation_errors"]:
                # Calibración: lo que no se puede respaldar con fuentes no se entrega como seguro.
                answer.citations = keep_valid_citations(answer, state["docs"])
                answer.confidence = "baja"
                if not answer.citations:
                    answer = Answer(answer="No pude respaldar una respuesta con citas verificables en los documentos.",
                                    answerable=False, confidence="baja", proposed_action=action)
            answer.answer, pii = redact_pii(answer.answer)
            if answer.proposed_action:
                p = answer.proposed_action.model_copy()
                p.title, _ = redact_pii(p.title)
                p.description, _ = redact_pii(p.description)
                answer.proposed_action = p
            extra.update(answerable=answer.answerable, confidence=answer.confidence, pii_redacted=pii)
            return {"answer": answer, "history": [{"question": state["safe_question"], "answer": answer.answer}]}

    def authorize_action(state: AgentState):
        # Permisos de herramienta: que el modelo proponga una acción no implica que el usuario pueda ejecutarla.
        with span("authorize_action", role=state["role"]) as extra:
            allowed = can_use_tool("crear_ticket", state["role"])
            extra.update(allowed=allowed)
            return {"action_status": "pendiente_aprobacion" if allowed else "denegada_por_permisos"}

    def human_review(state: AgentState):
        # El grafo se pausa aquí y queda persistido en el checkpointer hasta que una persona decide.
        requester = state.get("requested_by", "anonimo")
        decision = interrupt({"tool": "crear_ticket", "role": state["role"], "requested_by": requester,
                              "proposal": state["answer"].proposed_action.model_dump()})
        reviewer = decision.get("reviewer", "revisor")
        # Separación de funciones (four-eyes): quien solicita la acción no puede aprobarla.
        self_approval = reviewer == requester
        approved = bool(decision.get("approved")) and not self_approval
        with span("human_review", approved=approved, reviewer=reviewer, self_approval_blocked=self_approval):
            return {"action_status": "aprobada" if approved else "rechazada",
                    "ticket": {"approved_by": reviewer} if approved else None}

    def execute_action(state: AgentState, config: RunnableConfig):
        with span("execute_action") as extra:
            ticket = create_ticket(state["answer"].proposed_action, state["role"],
                                   requested_by=state.get("requested_by", "anonimo"),
                                   approved_by=state["ticket"]["approved_by"],
                                   thread_id=config["configurable"]["thread_id"])
            extra.update(ticket_id=ticket["id"])
            return {"action_status": "ejecutada", "ticket": ticket}

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

    def after_finalize(state: AgentState):
        return "authorize_action" if state["answer"].proposed_action else END

    def after_authorize(state: AgentState):
        return "human_review" if state["action_status"] == "pendiente_aprobacion" else END

    def after_review(state: AgentState):
        return "execute_action" if state["action_status"] == "aprobada" else END

    g = StateGraph(AgentState)
    for name, fn in [("guard_input", guard_input), ("retrieve", retrieve), ("generate", generate),
                     ("validate", validate), ("finalize", finalize), ("refuse", refuse), ("no_context", no_context),
                     ("authorize_action", authorize_action), ("human_review", human_review),
                     ("execute_action", execute_action)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["refuse", "retrieve"])
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "no_context"])
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, ["generate", "finalize"])
    g.add_conditional_edges("finalize", after_finalize, ["authorize_action", END])
    g.add_conditional_edges("authorize_action", after_authorize, ["human_review", END])
    g.add_conditional_edges("human_review", after_review, ["execute_action", END])
    for terminal in ("refuse", "no_context", "execute_action"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def _result(state: dict) -> dict:
    pending = state.get("__interrupt__")
    return {
        "answer": state["answer"],
        "action_status": state.get("action_status"),
        "pending_action": pending[0].value if pending else None,
        "ticket": state.get("ticket") if state.get("action_status") == "ejecutada" else None,
    }


def run(graph, question: str, role: str, thread_id: str = "default", user: str = "anonimo") -> dict:
    """Ejecuta un turno. Si hay una acción pendiente, `pending_action` trae lo que debe revisar una persona."""
    state = {"question": question, "role": role, "requested_by": user}
    return _result(graph.invoke(state, {"configurable": {"thread_id": thread_id}}))


def review(graph, thread_id: str, approved: bool, reviewer: str = "revisor") -> dict:
    """Reanuda un grafo pausado con la decisión humana."""
    state = graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer}),
                         {"configurable": {"thread_id": thread_id}})
    return _result(state)


def ask(graph, question: str, role: str, thread_id: str = "default") -> Answer:
    return run(graph, question, role, thread_id)["answer"]
