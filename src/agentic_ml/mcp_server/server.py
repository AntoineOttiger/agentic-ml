"""Serveur MCP exposant les 3 outils ML via transport stdio."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from agentic_ml.agents.training_agent.tools import get_dataset_profile, get_model_schema, launch_ml_pipeline
from agentic_ml.config import DEFAULT_N_TRIALS, DEFAULT_RANDOM_SEED

mcp = FastMCP("agentic-ml-tools")


@mcp.tool()
def get_dataset_profile_tool(run: str) -> dict[str, Any]:
    """Retourne les métadonnées complètes du dataset pour un run préparé donné."""
    return get_dataset_profile(run)


@mcp.tool()
def get_model_schema_tool(model_type: str) -> dict[str, Any]:
    """Retourne le schéma typé des hyperparamètres valides pour un type de modèle."""
    return get_model_schema(model_type)


@mcp.tool()
def launch_ml_pipeline_tool(
    prepared_run: str,
    model_type: str,
    search_space: dict[str, Any],
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Lance l'optimisation Optuna + entraînement pour une configuration donnée.

    Retourne train_f1, eval_f1, overfitting_score et best_hyperparams,
    ou {"error": "..."} si la configuration est invalide.
    """
    return launch_ml_pipeline(
        prepared_run,
        model_type,
        search_space,
        n_trials=n_trials,
        seed=seed,
    )
