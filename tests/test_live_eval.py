"""Prueba de regresión contra el modelo real. Opt-in: RUN_LIVE=1 pytest (consume cuota de la API)."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="prueba con modelo real: ejecutar con RUN_LIVE=1")


def test_golden_set_thresholds():
    from evals.run_eval import run

    s = run()
    assert s["fugas"] == 0
    assert s["decision_responder_correcta"] >= 0.9
    assert s["retrieval_recall_promedio"] >= 0.8
    assert s["citas_validas"] >= 0.8
