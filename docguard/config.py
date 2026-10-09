"""Configuración centralizada, leída desde variables de entorno (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DOCGUARD_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("DOCGUARD_INDEX_DIR", ROOT / "index"))
TRACE_FILE = Path(os.getenv("DOCGUARD_TRACE_FILE", ROOT / "logs" / "traces.jsonl"))

# "gemini" usa el SDK nativo; "openai_compat" usa cualquier gateway compatible con OpenAI
# (p. ej. un GenAI Gateway corporativo o el endpoint OpenAI-compatible de Gemini).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemini-3.5-flash-lite")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "2"))

TICKETS_FILE = Path(os.getenv("DOCGUARD_TICKETS_FILE", ROOT / "logs" / "tickets.jsonl"))
# Qué roles pueden usar cada herramienta. compliance es de solo lectura.
TOOL_PERMISSIONS = {"crear_ticket": {"analista", "crisis", "admin"}}

# USD por 1M de tokens, para estimar costo por consulta. Verificar con la lista de precios vigente.
PRICE_INPUT_PER_M = float(os.getenv("PRICE_INPUT_PER_M", "0.30"))
PRICE_OUTPUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_M", "2.50"))
