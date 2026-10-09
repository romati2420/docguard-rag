"""Pruebas de la API: autenticación, aislamiento por usuario y separación de funciones."""
import json

import pytest
from fastapi.testclient import TestClient

from docguard import api, config
from docguard.graph import build_graph
from docguard.retrieval import Retriever
from docguard.schemas import Answer, TicketProposal

from .test_graph import GOOD, FakeAnswerer

TOKENS = {"ana": "tok-ana", "carla": "tok-carla", "sofia": "tok-sofia", "admin": "tok-admin"}
TICKET = Answer(answer="Propongo un ticket.", answerable=False, confidence="alta",
                proposed_action=TicketProposal(title="Phishing", description="Correo sospechoso", priority="alta"))


@pytest.fixture
def client(tmp_path, monkeypatch, vectorstore):
    import hashlib

    from docguard import auth

    users = [("ana", "analista", False), ("carla", "compliance", False), ("sofia", "analista", True),
             ("admin", "admin", True)]
    f = tmp_path / "users.json"
    f.write_text(json.dumps({"users": [{"user": u, "role": r, "reviewer": rv,
                                        "token_sha256": hashlib.sha256(TOKENS[u].encode()).hexdigest()}
                                       for u, r, rv in users]}), encoding="utf-8")
    monkeypatch.setattr(config, "USERS_FILE", f)
    auth._load_users.cache_clear()
    for d in vectorstore.docstore._dict.values():
        d.metadata["allowed_roles"] = d.metadata["allowed_roles"] + ["compliance", "admin"]
    fake = FakeAnswerer(TICKET)
    graph = build_graph(Retriever(vectorstore, k=4), fake)
    monkeypatch.setattr(api, "get_agent", lambda: (graph, None))
    api.PENDING.clear()
    c = TestClient(api.app)
    c.fake = fake
    yield c
    auth._load_users.cache_clear()


def h(user):
    return {"Authorization": f"Bearer {TOKENS[user]}"}


def test_requires_token(client):
    assert client.post("/ask", json={"question": "hola"}).status_code == 401
    assert client.post("/ask", json={"question": "hola"}, headers={"Authorization": "Bearer falso"}).status_code == 401


def test_role_comes_from_token_not_from_body(client):
    client.fake.answers = [GOOD]
    r = client.post("/ask", json={"question": "¿Funciones?", "role": "admin"}, headers=h("carla"))
    assert r.status_code == 200
    assert client.get("/me", headers=h("carla")).json()["role"] == "compliance"


def test_non_reviewer_cannot_review(client):
    pending = client.post("/ask", json={"question": "Crea un ticket"}, headers=h("ana")).json()["pending_action"]
    r = client.post("/review", json={"review_id": pending["review_id"], "approved": True}, headers=h("ana"))
    assert r.status_code == 403


def test_reviewer_cannot_approve_own_request(client):
    pending = client.post("/ask", json={"question": "Crea un ticket"}, headers=h("sofia")).json()["pending_action"]
    assert pending["review_id"] not in [p["review_id"] for p in client.get("/reviews/pending", headers=h("sofia")).json()]
    r = client.post("/review", json={"review_id": pending["review_id"], "approved": True}, headers=h("sofia"))
    assert r.status_code == 403 and "Separación de funciones" in r.json()["detail"]


def test_reviewer_approves_other_users_request(client):
    pending = client.post("/ask", json={"question": "Crea un ticket"}, headers=h("ana")).json()["pending_action"]
    assert pending["requested_by"] == "ana"
    listed = client.get("/reviews/pending", headers=h("sofia")).json()
    assert [p["review_id"] for p in listed] == [pending["review_id"]]
    r = client.post("/review", json={"review_id": pending["review_id"], "approved": True}, headers=h("sofia")).json()
    assert r["ticket"]["approved_by"] == "sofia" and r["ticket"]["requested_by"] == "ana"
    assert client.get("/reviews/pending", headers=h("sofia")).json() == []


def test_threads_are_isolated_per_user(client):
    client.fake.answers = [GOOD]
    client.post("/ask", json={"question": "Primera", "thread_id": "t1"}, headers=h("ana"))
    client.post("/ask", json={"question": "Segunda", "thread_id": "t1"}, headers=h("carla"))
    assert client.fake.calls[1]["history"] == []  # carla no ve el historial de ana aunque use el mismo thread_id


def test_metrics_admin_only(client):
    assert client.get("/metrics", headers=h("ana")).status_code == 403
    assert client.get("/metrics", headers=h("admin")).status_code == 200
