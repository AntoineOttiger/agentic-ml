"""État partagé de l'agent, réinjecté à chaque tour de la boucle LangGraph.

Reflète la structure décrite dans `.claude/skills/agentic-archi.md` : mémoire de la
recherche (`trial_log`, `best_trial`), budget (`max_runs`/`runs_used`) et critères
d'arrêt (`target_f1`, `epsilon`, `patience`).
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class Trial(TypedDict):
    """Un essai enregistré dans le `trial_log`."""

    run_id: int
    hypothesis: str
    model_type: str
    hyperparameters: dict[str, Any]
    train_f1: float
    eval_f1: float


class AgentState(TypedDict, total=False):
    """État de la boucle agentique.

    `total=False` : les champs sont peuplés progressivement par les nœuds.
    """

    # Contexte injecté à l'initialisation (immuable pendant la run)
    dataset_profile: dict[str, Any]
    objective: str
    prepared_run: str

    # Budget
    max_runs: int
    runs_used: int

    # Mode et critères d'arrêt
    stop_mode: Literal["convergence", "agent"]
    target_f1: float
    epsilon: float
    patience: int

    # Paramètres d'optimisation transmis à la pipeline
    n_trials: int
    seed: int

    # Mémoire de la recherche
    trial_log: list[Trial]
    best_trial: Optional[Trial]

    # Pont entre propose_experiment et run_pipeline
    current_experiment: Optional[dict[str, Any]]
    last_error: Optional[str]

    # Décision du nœud evaluate_stop
    decision: Literal["continue", "stop"]
    stop_reason: Optional[str]
