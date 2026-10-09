# Gobernanza y evaluación de riesgos de DocGuard

Documento de evaluación de riesgos del sistema de IA: qué puede salir mal, qué control lo mitiga y qué
prueba demuestra que el control funciona. Basado en OWASP Top 10 for LLM Applications (2025).

## Matriz de riesgos

| # | Riesgo | OWASP LLM | Impacto | Control implementado | Evidencia (prueba) |
|---|---|---|---|---|---|
| R1 | Prompt injection directa: el usuario intenta cambiar las instrucciones | LLM01 | Alto | `guard_input` bloquea patrones conocidos **antes** de llamar al modelo | `test_direct_injection_is_blocked_before_llm`, caso `injection_directa` |
| R2 | Prompt injection indirecta: un documento trae instrucciones incrustadas | LLM01 | Alto | Las líneas con instrucciones se eliminan del fragmento antes del prompt; el contexto se delimita como *datos* | `test_indirect_injection_chunk_is_quarantined`, caso `injection_indirecta` |
| R3 | Exposición de datos personales (Ley 21.719) | LLM02 | Alto | Redacción de RUT (con dígito verificador), email, teléfono y tarjetas (Luhn) en la pregunta, la respuesta y las propuestas de acción | `test_redacts_pii`, `test_pii_is_redacted_from_question_and_answer`, `test_pending_proposal_has_pii_redacted` |
| R4 | Acceso a documentos fuera del rol del usuario | LLM02 / LLM08 | Alto | Filtro por rol **dentro de la búsqueda** vectorial; sin entrada en el ACL, solo `admin` | `test_role_filter_hides_restricted_documents`, caso `crisis_sin_permiso` |
| R5 | Alucinación / respuesta sin respaldo | LLM09 | Medio | Salida estructurada + verificación literal de citas + reintento con retroalimentación + abstención calibrada | `test_invalid_citation_triggers_retry_with_feedback`, `test_persistently_unsupported_answer_is_downgraded`, caso `fuera_de_dominio` |
| R6 | Agencia excesiva: el agente ejecuta acciones con impacto externo | LLM06 | Alto | El modelo solo **propone**; permisos por herramienta + **aprobación humana** (`interrupt` de LangGraph) antes de ejecutar | `test_action_pauses_for_human_review`, `test_rejected_action_is_not_executed`, `test_read_only_role_cannot_use_tool` |
| R7 | Fuga de las instrucciones del sistema | LLM07 | Bajo | Regla explícita en el prompt + bloqueo de solicitudes de "system prompt" | caso `injection_indirecta` (sin fugas) |
| R8 | Consumo descontrolado / costo | LLM10 | Medio | Timeout por llamada, máximo de reintentos, registro de tokens y costo por nodo | `/metrics`, `logs/traces.jsonl` |
| R9 | Indisponibilidad del proveedor del modelo | — | Medio | Modelo de respaldo automático y reintentos ante cuota (429) en la ingesta | Verificado en la práctica: errores 503 del modelo principal resueltos por el respaldo |

## Supervisión humana

- Toda acción con impacto fuera del sistema (crear un ticket) se pausa en el nodo `human_review`. El estado queda
  persistido en el checkpointer y el grafo se reanuda solo con una decisión explícita (`POST /review`).
- La decisión registra quién aprobó (`approved_by`), y el ticket guarda el rol solicitante y el `thread_id` para
  trazabilidad.

## Monitoreo posterior al despliegue

`GET /metrics` agrega las trazas en indicadores operativos: consultas, bloqueos por injection, abstenciones,
respuestas de confianza baja, reintentos por citas, líneas en cuarentena, acciones aprobadas/rechazadas/denegadas,
tokens, costo estimado y latencia por nodo. Una subida en abstenciones o en reintentos es señal de degradación del
retrieval o del modelo.

## Ciclo de cambios

1. Todo cambio (prompt, modelo, segmentación) corre `pytest` (offline) y `python -m evals.run_eval` (modelo real).
2. Umbrales de regresión en `tests/test_live_eval.py`: 0 fugas, ≥90% de decisiones correctas, ≥80% de recall de
   recuperación y de citas válidas.
3. Cada falla encontrada se convierte en un caso del set de referencia o en una prueba unitaria.

## Limitaciones conocidas

- Los guardrails de injection son heurísticos: bloquean patrones comunes, no ataques nuevos u ofuscados. La
  siguiente capa sería un clasificador dedicado.
- El set de referencia tiene 10 casos: sirve como regresión, no como benchmark estadístico.
- El rol llega en la petición; en producción debe venir del token de identidad (IAM / OIDC).
- El sistema de tickets es local (JSONL) y simula una API tipo Jira.
