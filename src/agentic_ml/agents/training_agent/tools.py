"""Tools de l'agent (fonctions pures).

Trois des quatre tools décrits dans `agentic-archi.md` ; le RAG
(`query_ml_knowledge_base`) est hors périmètre de cette première version.

Les tools sont des fonctions ordinaires : `propose_experiment` les utilise pour
construire son contexte, `run_pipeline` appelle `launch_ml_pipeline`. Elles
restent promouvables en `@tool` LangChain sans changement de signature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_ml.config import DEFAULT_N_TRIALS, DEFAULT_RANDOM_SEED
from agentic_ml.training.models import available_models, get_model_spec
from agentic_ml.training.optimizer import HyperparameterOptimizer
from agentic_ml.training.search_space import validate_search_space

from agentic_ml.agents.training_agent.profile import build_dataset_profile


def get_dataset_profile(run: str | Path) -> dict[str, Any]:
    """Renvoie en un seul appel l'ensemble des métadonnées du dataset."""
    return build_dataset_profile(run)


def get_model_schema(model_type: str) -> dict[str, Any]:
    """Renvoie le schéma typé des hyperparamètres valides d'un modèle."""
    if model_type not in available_models():
        return {
            "error": f"Modèle inconnu '{model_type}'.",
            "available_models": available_models(),
        }
    return {
        "model_type": model_type,
        "needs_label_encoding": get_model_spec(model_type).needs_label_encoding,
        "hyperparameters": get_model_spec(model_type).default_search_space,
    }


def _validate(model_type: str, search_space: dict[str, dict]) -> str | None:
    """Valide modèle, structure et noms d'hyperparamètres. Renvoie une erreur ou None."""
    if model_type not in available_models():
        return (
            f"Modèle inconnu '{model_type}'. Disponibles : {available_models()}."
        )
    try:
        validate_search_space(search_space)
    except ValueError as exc:
        return str(exc)

    allowed = set(get_model_spec(model_type).default_search_space)
    unknown = [name for name in search_space if name not in allowed]
    if unknown:
        return (
            f"Hyperparamètres inconnus pour '{model_type}' : {unknown}. "
            f"Autorisés : {sorted(allowed)}."
        )
    return None


def launch_ml_pipeline(
    prepared_run: str | Path,
    model_type: str,
    search_space: dict[str, dict],
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Lance la pipeline d'optimisation ML pour une config donnée.

    Valide les hyperparamètres au préalable : en cas de nom/borne invalide,
    renvoie une erreur structurée `{"error": ...}` que l'agent peut lire et
    corriger plutôt qu'un crash. Sinon renvoie scores et meilleurs hyperparams.
    """
    error = _validate(model_type, search_space)
    if error is not None:
        return {"error": error}

    optimizer = HyperparameterOptimizer(random_seed=seed, n_trials=n_trials)
    result = optimizer.optimize(prepared_run, model_type, search_space)

    return {
        "train_f1": result.train_f1,
        "eval_f1": result.eval_f1,
        "overfitting_score": result.train_f1 - result.eval_f1,
        "best_hyperparams": result.best_params,
    }
