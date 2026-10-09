"""Herramientas del agente. El sistema de tickets es local (JSONL) y simula una API tipo Jira."""
import json
from datetime import datetime, timezone

from . import config
from .schemas import TicketProposal


def can_use_tool(tool: str, role: str) -> bool:
    return role in config.TOOL_PERMISSIONS.get(tool, set())


def create_ticket(proposal: TicketProposal, role: str, approved_by: str, thread_id: str,
                  requested_by: str = "anonimo") -> dict:
    """Crea el ticket. Solo se llama después de la aprobación humana."""
    config.TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = config.TICKETS_FILE.read_text(encoding="utf-8").splitlines() if config.TICKETS_FILE.exists() else []
    ticket = {
        "id": f"DG-{len(existing) + 1}",
        **proposal.model_dump(),
        "requested_by": requested_by,
        "requested_by_role": role,
        "approved_by": approved_by,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with config.TICKETS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return ticket
