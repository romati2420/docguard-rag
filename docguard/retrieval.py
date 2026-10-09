"""Recuperación con filtro de permisos: un rol solo ve los documentos que su ACL autoriza."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import config


class Retriever:
    def __init__(self, vectorstore: FAISS, k: int = config.TOP_K, fetch_k: int = 50):
        self.vs = vectorstore
        self.k = k
        self.fetch_k = fetch_k

    def __call__(self, question: str, role: str) -> list[Document]:
        # El filtro se aplica dentro de la búsqueda, no después: los fragmentos no autorizados
        # nunca llegan al prompt del modelo.
        return self.vs.similarity_search(
            question,
            k=self.k,
            fetch_k=self.fetch_k,
            filter=lambda md: role in md.get("allowed_roles", []),
        )
