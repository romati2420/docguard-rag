"""CLI: python -m docguard ingest | ask "pregunta" [--role analista]"""
import argparse

from . import config
from .graph import ask
from .ingest import build_index
from .llm import get_embeddings


def main():
    parser = argparse.ArgumentParser(prog="docguard")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Procesa data/ y construye el índice FAISS")
    p_ask = sub.add_parser("ask", help="Hace una pregunta al agente")
    p_ask.add_argument("question")
    p_ask.add_argument("--role", default="analista")
    p_ask.add_argument("--thread", default="cli")
    args = parser.parse_args()

    if args.cmd == "ingest":
        vs = build_index(get_embeddings())
        print(f"Índice creado en {config.INDEX_DIR} con {vs.index.ntotal} fragmentos.")
        return

    from .app import get_agent

    graph, _ = get_agent()
    answer = ask(graph, args.question, args.role, args.thread)
    print(f"\n{answer.answer}\n")
    print(f"answerable={answer.answerable} · confianza={answer.confidence}")
    for c in answer.citations:
        print(f"  [{c.source} p.{c.page}] \"{c.quote}\"")


if __name__ == "__main__":
    main()
