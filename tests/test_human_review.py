"""Supervisión humana: ninguna acción con impacto externo se ejecuta sin aprobación y permisos."""
from docguard import config
from docguard.graph import build_graph, review, run
from docguard.retrieval import Retriever
from docguard.schemas import Answer, TicketProposal

from .test_graph import FakeAnswerer

PROPOSAL = Answer(
    answer="Propongo crear un ticket para escalar el incidente.", answerable=False, confidence="alta",
    proposed_action=TicketProposal(title="Phishing reportado por cliente 11.111.111-1",
                                   description="Correo sospechoso recibido por juan@empresa.cl", priority="alta"),
)


def _tickets():
    return config.TICKETS_FILE.read_text(encoding="utf-8").splitlines() if config.TICKETS_FILE.exists() else []


def test_action_pauses_for_human_review(vectorstore):
    graph = build_graph(Retriever(vectorstore, k=4), FakeAnswerer(PROPOSAL))
    result = run(graph, "Crea un ticket por un phishing", role="analista", thread_id="hr1")
    assert result["action_status"] == "pendiente_aprobacion"
    assert result["pending_action"]["tool"] == "crear_ticket"
    assert _tickets() == []  # nada se ejecuta antes de la aprobación


def test_pending_proposal_has_pii_redacted(vectorstore):
    graph = build_graph(Retriever(vectorstore, k=4), FakeAnswerer(PROPOSAL))
    proposal = run(graph, "Crea un ticket", role="analista", thread_id="hr2")["pending_action"]["proposal"]
    assert "11.111.111-1" not in proposal["title"] and "juan@empresa.cl" not in proposal["description"]


def test_approved_action_is_executed_once(vectorstore):
    graph = build_graph(Retriever(vectorstore, k=4), FakeAnswerer(PROPOSAL))
    run(graph, "Crea un ticket por un phishing", role="analista", thread_id="hr3")
    result = review(graph, "hr3", approved=True, reviewer="supervisora.soc")
    assert result["action_status"] == "ejecutada"
    assert result["ticket"]["id"] == "DG-1" and result["ticket"]["approved_by"] == "supervisora.soc"
    assert len(_tickets()) == 1


def test_rejected_action_is_not_executed(vectorstore):
    graph = build_graph(Retriever(vectorstore, k=4), FakeAnswerer(PROPOSAL))
    run(graph, "Crea un ticket por un phishing", role="analista", thread_id="hr4")
    result = review(graph, "hr4", approved=False)
    assert result["action_status"] == "rechazada" and result["ticket"] is None
    assert _tickets() == []


def test_read_only_role_cannot_use_tool(vectorstore, docs):
    for d in docs:
        d.metadata["allowed_roles"].append("compliance")
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import DeterministicFakeEmbedding

    vs = FAISS.from_documents(docs, DeterministicFakeEmbedding(size=64))
    graph = build_graph(Retriever(vs, k=4), FakeAnswerer(PROPOSAL))
    result = run(graph, "Crea un ticket por un phishing", role="compliance", thread_id="hr6")
    assert result["action_status"] == "denegada_por_permisos"
    assert result["pending_action"] is None and _tickets() == []


def test_graph_blocks_self_approval_even_if_api_is_bypassed(vectorstore):
    graph = build_graph(Retriever(vectorstore, k=4), FakeAnswerer(PROPOSAL))
    run(graph, "Crea un ticket por un phishing", role="analista", thread_id="hr7", user="ana")
    result = review(graph, "hr7", approved=True, reviewer="ana")
    assert result["action_status"] == "rechazada" and _tickets() == []
