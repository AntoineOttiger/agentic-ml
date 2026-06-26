"""Persistance d'une run d'agent : trial_log, résumé et best model.

Écrit dans `results/agent_runs/<prepared_run>/<timestamp>/` :
- `trial_log.json` : l'historique complet des essais (train_f1/eval_f1 par run) ;
- `summary.json`   : meilleur essai + configuration de la run + motif d'arrêt ;
- `best_model.joblib` : le best model réentraîné sur le train (avec son LabelEncoder).

Le best model est *réentraîné* ici une seule fois à partir du `best_trial` : aucun
estimateur ne transite par le state LangGraph, qui reste JSON-sérialisable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib

from agentic_ml.config import DEFAULT_RANDOM_SEED, RESULTS_DIR
from agentic_ml.training.dataset import load_prepared_run
from agentic_ml.training.models import build_estimator

from agentic_ml.agents.training_agent.state import AgentState

logger = logging.getLogger("agentic_ml.agents.training_agent")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_best_model(state: AgentState, out_dir: Path) -> None:
    """Réentraîne le best model sur le train et le sérialise avec son encodeur."""
    best = state.get("best_trial")
    if best is None:
        logger.info("persist | aucun essai réussi : best_model.joblib non écrit")
        return

    data = load_prepared_run(state["prepared_run"])
    estimator = build_estimator(
        best["model_type"], best["hyperparameters"], random_state=state.get("seed", DEFAULT_RANDOM_SEED)
    )
    estimator.fit(data.X_train, data.y_train)

    joblib.dump(
        {
            "model": estimator,
            "label_encoder": data.label_encoder,
            "feature_columns": data.feature_columns,
            "model_type": best["model_type"],
            "hyperparameters": best["hyperparameters"],
        },
        out_dir / "best_model.joblib",
    )


def persist_results(state: AgentState, *, results_dir: Path = RESULTS_DIR) -> Path:
    """Écrit les artefacts de la run et renvoie le dossier créé."""
    prepared_run = state.get("prepared_run") or "unknown"
    out_dir = results_dir / prepared_run / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "trial_log.json", state.get("trial_log", []))
    _write_json(
        out_dir / "summary.json",
        {
            "best_trial": state.get("best_trial"),
            "runs_used": state.get("runs_used", 0),
            "max_runs": state["max_runs"],
            "stop_mode": state.get("stop_mode"),
            "stop_reason": state.get("stop_reason"),
            "objective": state.get("objective"),
            "target_f1": state.get("target_f1"),
            "prepared_run": state.get("prepared_run"),
            "seed": state.get("seed"),
        },
    )
    _save_best_model(state, out_dir)

    logger.info("persist | résultats écrits dans %s", out_dir)
    return out_dir
