"""Ingesta: extracción de PDF/Markdown, segmentación y construcción del índice FAISS."""
import json
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # palabras cortadas al final de línea
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def load_acl(data_dir: Path) -> dict[str, list[str]]:
    acl_file = data_dir / "acl.json"
    if not acl_file.exists():
        return {}
    return {k: v for k, v in json.loads(acl_file.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    """Una Document por página (PDF) o por archivo (Markdown/texto), con metadatos de fuente y ACL."""
    acl = load_acl(data_dir)
    docs: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        # Sin entrada en el ACL, el documento queda restringido a admin (denegar por defecto).
        roles = acl.get(path.name, ["admin"])
        if path.suffix.lower() == ".pdf":
            for i, page in enumerate(PdfReader(path).pages, start=1):
                text = _clean(page.extract_text() or "")
                if text:
                    docs.append(Document(text, metadata={"source": path.name, "page": i, "allowed_roles": roles}))
        elif path.suffix.lower() in {".md", ".txt"}:
            docs.append(Document(_clean(path.read_text(encoding="utf-8")),
                                 metadata={"source": path.name, "page": 1, "allowed_roles": roles}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n#", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def _with_backoff(fn, retries: int = 5, wait_s: float = 30.0):
    """Reintenta ante límites de cuota (HTTP 429) con espera creciente."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # el SDK envuelve el 429 en distintos tipos de excepción
            if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e) or attempt == retries - 1:
                raise
            print(f"  cuota alcanzada, reintentando en {wait_s:.0f}s...")
            time.sleep(wait_s)
            wait_s *= 1.5


def build_index(embeddings: Embeddings, data_dir: Path = config.DATA_DIR,
                index_dir: Path | None = config.INDEX_DIR, batch_size: int = config.EMBED_BATCH_SIZE) -> FAISS:
    """Construye el índice por lotes para respetar los límites de cuota del proveedor de embeddings."""
    chunks = split_documents(load_documents(data_dir))
    vs = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vs is None:
            vs = _with_backoff(lambda: FAISS.from_documents(batch, embeddings))
        else:
            _with_backoff(lambda: vs.add_documents(batch))
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} fragmentos indexados")
    if index_dir:
        vs.save_local(str(index_dir))
    return vs


def load_index(embeddings: Embeddings, index_dir: Path = config.INDEX_DIR) -> FAISS:
    # El índice usa pickle: cargar solo índices generados localmente por este proyecto.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
