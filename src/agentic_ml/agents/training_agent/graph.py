"""Assemblage du StateGraph et point d'entrée `run_agent`."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agentic_ml.config import (
    AGENT_MODEL,
    CONVERGENCE_EPSILON,
    CONVERGENCE_PATIENCE,
    DEFAULT_MAX_RUNS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STOP_MODE,
    DEFAULT_TARGET_F1,
)

from agentic_ml.agents.training_agent.nodes import (
    evaluate_stop,
    make_llm,
    propose_experiment,
    route_after_eval,
    run_pipeline,
)
from agentic_ml.agents.training_agent.profile import latest_prepared_run
from agentic_ml.agents.training_agent.results import persist_results
from agentic_ml.agents.training_agent.state import AgentState
from agentic_ml.agents.training_agent.tools import get_dataset_profile


def build_agent_graph(llm: BaseChatModel):
    """Construit et compile le graphe de la boucle de recherche.

    Flux : propose_experiment → run_pipeline → evaluate_stop → (boucle | fin).
    """
    graph = StateGraph(AgentState)

    graph.add_node("propose_experiment", partial(propose_experiment, llm=llm))
    graph.add_node("run_pipeline", run_pipeline)
    graph.add_node("evaluate_stop", partial(evaluate_stop, llm=llm))

    graph.add_edge(START, "propose_experiment")
    graph.add_edge("propose_experiment", "run_pipeline")
    graph.add_edge("run_pipeline", "evaluate_stop")
    graph.add_conditional_edges(
        "evaluate_stop",
        route_after_eval,
        {"continue": "propose_experiment", "stop": END},
    )

    return graph.compile()


def run_agent(
    *,
    prepared_run: Optional[str] = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    target_f1: float = DEFAULT_TARGET_F1,
    stop_mode: str = DEFAULT_STOP_MODE,
    model: str = AGENT_MODEL,
    seed: int = DEFAULT_RANDOM_SEED,
    results_dir: Optional[Path] = None,
) -> AgentState:
    """Exécute la boucle agentique de bout en bout et renvoie l'état final.

    Args:
        prepared_run: identifiant d'un run de `data/02_prepared` (défaut : le plus récent).
        stop_mode: "convergence" (arrêt déterministe) ou "agent" (le LLM décide,
            `max_runs` restant le seul garde-fou dur).
        results_dir: dossier de persistance (défaut : `config.RESULTS_DIR`).

    Les résultats (trial_log, summary, best model) sont écrits sur disque à la fin.
    """
    run_id = prepared_run or latest_prepared_run()

    initial_state: AgentState = {
        "dataset_profile": get_dataset_profile(run_id),
        "objective": "maximize eval_f1",
        "prepared_run": run_id,
        "agent_model": model,
        "max_runs": max_runs,
        "runs_used": 0,
        "stop_mode": stop_mode,
        "target_f1": target_f1,
        "epsilon": CONVERGENCE_EPSILON,
        "patience": CONVERGENCE_PATIENCE,
        "seed": seed,
        "trial_log": [],
        "best_trial": None,
        "current_experiment": None,
        "last_error": None,
        "decision": "continue",
        "stop_reason": None,
    }

    app = build_agent_graph(make_llm(model))
    # Plafond de sécurité : au pire un propose+run+eval par budget, plus une marge.
    final_state: AgentState = app.invoke(
        initial_state, {"recursion_limit": max_runs * 3 + 10}
    )

    persist_kwargs = {} if results_dir is None else {"results_dir": results_dir}
    persist_results(final_state, **persist_kwargs)
    return final_state
