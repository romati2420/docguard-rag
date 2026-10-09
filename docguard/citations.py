"""Verificación de citas: cada cita debe existir literalmente en un fragmento recuperado."""
import re
import unicodedata

from langchain_core.documents import Document

from .schemas import Answer, Citation


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[“”\"'«»]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str, min_token_overlap: float = 0.85) -> bool:
    q, t = _norm(quote), _norm(text)
    if q in t:
        return True
    # Tolerancia a diferencias menores de extracción del PDF (guiones, espacios, tildes).
    q_tokens = q.split()
    t_tokens = set(t.split())
    return bool(q_tokens) and sum(tok in t_tokens for tok in q_tokens) / len(q_tokens) >= min_token_overlap


def check_citation(c: Citation, docs: list[Document]) -> str | None:
    """Devuelve el motivo del error, o None si la cita es válida."""
    candidates = [d for d in docs if d.metadata["source"] == c.source and d.metadata["page"] == c.page]
    if not candidates:
        return f"La cita a {c.source} p.{c.page} no corresponde a ningún fragmento entregado."
    if not any(quote_in_text(c.quote, d.page_content) for d in candidates):
        return f"El texto citado de {c.source} p.{c.page} no aparece literalmente en el fragmento: '{c.quote[:80]}'"
    return None


def validate_answer(answer: Answer, docs: list[Document]) -> list[str]:
    if not answer.answerable:
        return []
    if not answer.citations:
        # Una solicitud de acción (ticket) puede no consultar documentos; se controla en la aprobación humana.
        return [] if answer.proposed_action else ["La respuesta no incluye citas a las fuentes."]
    return [err for c in answer.citations if (err := check_citation(c, docs))]


def keep_valid_citations(answer: Answer, docs: list[Document]) -> list[Citation]:
    return [c for c in answer.citations if check_citation(c, docs) is None]
