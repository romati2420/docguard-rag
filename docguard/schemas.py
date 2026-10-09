"""Contratos de salida estructurada validados con Pydantic."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUOTE_CHARS = 300


class Citation(BaseModel):
    source: str = Field(description="Nombre exacto del archivo fuente, tal como aparece en [fuente: ...]")
    page: int = Field(ge=1, description="Número de página indicado en [página: ...]")
    quote: str = Field(
        min_length=1,
        description="Cita textual y literal (copiada sin cambios) del fragmento que respalda la respuesta, máximo 300 caracteres",
    )

    @field_validator("quote", mode="before")
    @classmethod
    def truncate_quote(cls, v: str) -> str:
        # Una cita larga no invalida la respuesta: se recorta en un límite de palabra y sigue
        # siendo un fragmento literal de la fuente, por lo que la verificación de citas sigue aplicando.
        v = str(v).strip()
        if len(v) <= MAX_QUOTE_CHARS:
            return v
        return v[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0]


class TicketProposal(BaseModel):
    """Acción propuesta por el agente. Nunca se ejecuta sin aprobación humana."""
    title: str = Field(max_length=120, description="Título breve del ticket")
    description: str = Field(description="Descripción del problema o solicitud, sin datos personales")
    priority: Literal["baja", "media", "alta", "critica"] = Field(description="Prioridad sugerida")


class Answer(BaseModel):
    answer: str = Field(description="Respuesta en español basada solo en los documentos entregados")
    citations: list[Citation] = Field(default_factory=list, description="Fuentes que respaldan la respuesta")
    answerable: bool = Field(description="false si los documentos no contienen información suficiente")
    confidence: Literal["alta", "media", "baja"] = Field(description="Confianza en que la respuesta está respaldada")
    proposed_action: TicketProposal | None = Field(
        default=None,
        description="Solo si el usuario pide explícitamente crear/registrar/escalar un ticket; en otro caso null",
    )
