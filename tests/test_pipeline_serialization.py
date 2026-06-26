"""Test de non-régression : la sortie de `launch_ml_pipeline` doit être
entièrement JSON-sérialisable (aucun type numpy résiduel).

Reproduit le bug « Unable to serialize unknown type: numpy.int64 » : `json.dumps`
échoue sur tout scalaire numpy non converti. Le garde-fou réel est `_to_python()`
dans `tools.py`. On lance une petite optimisation Optuna (n_trials faible) pour
rester rapide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# src/ sur le path (le projet n'installe pas le package en editable dans les tests).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_ml.agents.training_agent.tools import launch_ml_pipeline

# creditcard : seul run compatible avec config.TARGET_COL ("Class"). Ses labels
# sont des entiers (0/1) — c'est précisément ce qui produisait des clés
# numpy.int64 non sérialisables dans le rapport par classe.
PREPARED_RUN = "creditcard_001_001"
MODEL_TYPE = "random_forest"
SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 50, "high": 60, "step": 10},
    "max_depth": {"type": "int", "low": 3, "high": 5},
}
EXPECTED_CLASSES = ("0", "1")


def _assert_class_report(result: dict) -> None:
    """Vérifie la présence, la forme et la sérialisabilité du rapport par classe."""
    assert "val_class_report" in result, "val_class_report absent du résultat"
    report = result["val_class_report"]
    # Clés de classe : strings (labels stringifiés), jamais des scalaires numpy.
    for cls in EXPECTED_CLASSES:
        assert cls in report, f"classe '{cls}' absente du rapport : {list(report)}"
        assert {"precision", "recall", "f1-score", "support"} <= set(report[cls])
    assert all(isinstance(k, str) for k in report), (
        f"clés non-string dans le rapport : {[type(k).__name__ for k in report]}"
    )
    assert "accuracy" in report
    assert "macro avg" in report


def test_launch_ml_pipeline_is_json_serializable() -> None:
    result = launch_ml_pipeline(
        PREPARED_RUN, MODEL_TYPE, SEARCH_SPACE, n_trials=3, seed=42
    )
    assert "error" not in result, result.get("error")
    # json.dumps lève TypeError sur tout numpy.int64/float64 résiduel.
    json.dumps(result)
    _assert_class_report(result)


if __name__ == "__main__":
    # Exécutable sans pytest : lance chaque test et rapporte le résultat.
    failures = 0
    for name, fn in [
        ("test_launch_ml_pipeline_is_json_serializable", test_launch_ml_pipeline_is_json_serializable),
    ]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 — rapport de test
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
