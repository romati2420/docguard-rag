"""Autenticación del prototipo: token Bearer → usuario, rol y permiso de revisión, resueltos en el servidor.

El rol nunca se acepta desde la petición. Los tokens se guardan como hash SHA-256 y se comparan en tiempo
constante. En producción esto se reemplaza por un token verificado de IAM / OIDC.
"""
import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Header, HTTPException

from . import config


@dataclass(frozen=True)
class User:
    name: str
    role: str
    reviewer: bool


@lru_cache(maxsize=1)
def _load_users() -> tuple[dict, ...]:
    return tuple(json.loads(config.USERS_FILE.read_text(encoding="utf-8"))["users"])


def authenticate(token: str) -> User | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    for u in _load_users():
        if hmac.compare_digest(digest, u["token_sha256"]):
            return User(u["user"], u["role"], bool(u.get("reviewer")))
    return None


def current_user(authorization: str = Header(default="")) -> User:
    """Dependencia de FastAPI: exige `Authorization: Bearer <token>` válido."""
    scheme, _, token = authorization.partition(" ")
    user = authenticate(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
    if user is None:
        raise HTTPException(status_code=401, detail="Token inválido o ausente",
                            headers={"WWW-Authenticate": "Bearer"})
    return user
