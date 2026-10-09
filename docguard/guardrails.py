"""Guardrails: detección de prompt injection y protección de datos personales (PII).

Son heurísticas deterministas y baratas que corren antes y después del LLM. No reemplazan
un clasificador dedicado, pero bloquean los patrones más comunes sin costo de tokens.
"""
import re
import unicodedata

_INJECTION_PATTERNS = [
    r"ignor(a|e|ar)\b.{0,40}\b(instrucciones|indicaciones|reglas|instructions|rules)",
    r"(olvida|forget)\b.{0,40}\b(instrucciones|todo|instructions|everything)",
    r"(revela|muestra|imprime|reveal|print|show)\b.{0,40}\b(system prompt|prompt del sistema|instrucciones del sistema)",
    r"\bsystem prompt\b",
    r"\b(ahora eres|act[uú]a como|you are now|pretend to be)\b",
    r"\b(jailbreak|modo desarrollador|developer mode|DAN)\b",
    r"</?(system|instrucciones|documentos)>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RUT_RE = re.compile(r"\b(\d{1,2}\.?\d{3}\.?\d{3})-?([\dkK])\b")
_PHONE_CL_RE = re.compile(r"(?:\+?56\s?)?9\s?\d{4}\s?\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def detect_injection(text: str) -> str | None:
    """Devuelve el patrón detectado o None si el texto parece seguro."""
    norm = _normalize(text)
    for rx in _INJECTION_RE:
        if m := rx.search(norm):
            return m.group(0)
    return None


def strip_injected_lines(text: str) -> tuple[str, int]:
    """Elimina solo las líneas con instrucciones incrustadas; conserva el contenido legítimo del fragmento."""
    kept, removed = [], 0
    for line in text.splitlines():
        if detect_injection(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept).strip(), removed


def rut_is_valid(body: str, dv: str) -> bool:
    """Valida el dígito verificador de un RUT chileno (módulo 11)."""
    digits = body.replace(".", "")
    total, factor = 0, 2
    for d in reversed(digits):
        total += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    expected = 11 - total % 11
    expected_dv = {11: "0", 10: "K"}.get(expected, str(expected))
    return expected_dv == dv.upper()


def luhn_is_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Enmascara PII y devuelve (texto_redactado, tipos_encontrados)."""
    found: list[str] = []

    def sub(rx, label, validator=None):
        nonlocal text

        def repl(m):
            if validator and not validator(m):
                return m.group(0)
            found.append(label)
            return f"[{label} REDACTADO]"

        text = rx.sub(repl, text)

    sub(_EMAIL_RE, "EMAIL")
    sub(_RUT_RE, "RUT", lambda m: rut_is_valid(m.group(1), m.group(2)))
    sub(_CARD_RE, "TARJETA", lambda m: luhn_is_valid(m.group(0)))
    sub(_PHONE_CL_RE, "TELEFONO")
    return text, found
