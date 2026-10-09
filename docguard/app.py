"""Ensamblado del agente con los componentes reales (Gemini + índice FAISS en disco)."""
from functools import lru_cache

from . import config
from .graph import build_graph
from .ingest import build_index, load_index
from .llm import get_embeddings, make_answer_fn
from .retrieval import Retriever


@lru_cache(maxsize=1)
def get_agent():
    embeddings = get_embeddings()
    if (config.INDEX_DIR / "index.faiss").exists():
        vs = load_index(embeddings)
    else:
        vs = build_index(embeddings)
    retriever = Retriever(vs)
    return build_graph(retriever, make_answer_fn()), retriever
