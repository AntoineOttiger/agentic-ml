"""Nœuds de la boucle agentique.

Seul `propose_experiment` mobilise le LLM. `run_pipeline` et `evaluate_stop` sont
déterministes — l'auto-terminaison est volontairement séparée de la proposition.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ml.config import AGENT_MODEL, AGENT_PROVIDER, MAX_N_TRIALS, MIN_N_TRIALS
from agentic_ml.utils.rate_limiter import RateLimitCallback, get_rate_limiter

from agentic_ml.agents.training_agent.prompts import (
    STOP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context,
    build_stop_context,
)
from agentic_ml.agents.training_agent.report import format_class_report
from agentic_ml.agents.training_agent.schema import Experiment, StopDecision, to_search_space
from agentic_ml.agents.training_agent.state import AgentState, Trial
from agentic_ml.agents.training_agent.tools import launch_ml_pipeline

logger = logging.getLogger("agentic_ml.agents.training_agent")


def make_llm(model: str = AGENT_MODEL, *, temperature: float = 0.4) -> BaseChatModel:
    """Instancie le client LLM selon AGENT_PROVIDER avec rate limiting."""
    callback = RateLimitCallback(get_rate_limiter())
    if AGENT_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, callbacks=[callback])
    from langchain_mistralai import BaseChatModel
    return BaseChatModel(model=model, temperature=temperature, callbacks=[callback])


def propose_experiment(state: AgentState, llm: BaseChatModel) -> dict:
    """Formule une hypothèse et propose une configuration (model_type, search_space)."""
    structured = llm.with_structured_output(Experiment)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_context(state)),
    ]
    logger.info("propose | appel Mistral en cours (essai %d/%d)…", state.get("runs_used", 0) + 1, state["max_runs"])
    experiment: Experiment = structured.invoke(messages)

    current = {
        "hypothesis": experiment.hypothesis,
        "model_type": experiment.model_type,
        "search_space": to_search_space(experiment.search_space),
        "n_trials": experiment.n_trials,
    }
    logger.info(
        "propose | modèle=%s | hypothèse=%s", current["model_type"], current["hypothesis"]
    )
    return {"current_experiment": current, "last_error": None}


def run_pipeline(state: AgentState) -> dict:
    """Valide et lance la pipeline pour la config proposée, met à jour la mémoire."""
    exp = state["current_experiment"]
    # Garde-fou : la valeur choisie par l'agent est clampée dans les bornes config.
    n_trials = max(MIN_N_TRIALS, min(MAX_N_TRIALS, exp["n_trials"]))
    result = launch_ml_pipeline(
        state["prepared_run"],
        exp["model_type"],
        exp["search_space"],
        n_trials=n_trials,
        seed=state["seed"],
    )

    if "error" in result:
        logger.warning("run | configuration rejetée : %s", result["error"])
        return {"current_experiment": None, "last_error": result["error"]}

    trial: Trial = {
        "run_id": state.get("runs_used", 0) + 1,
        "hypothesis": exp["hypothesis"],
        "model_type": exp["model_type"],
        "hyperparameters": result["best_hyperparams"],
        "n_trials": n_trials,
        "train_f1": result["train_f1"],
        "eval_f1": result["eval_f1"],
        "val_class_report": result["val_class_report"],
    }

    trial_log = state.get("trial_log", []) + [trial]
    best = state.get("best_trial")
    if best is None or trial["eval_f1"] > best["eval_f1"]:
        best = trial

    logger.info(
        "run | essai #%d %s | train_f1=%.4f eval_f1=%.4f (overfit=%.4f)",
        trial["run_id"], trial["model_type"], trial["train_f1"], trial["eval_f1"],
        result["overfitting_score"],
    )
    logger.info(
        "run | détail par classe (validation) :\n%s",
        format_class_report(result["val_class_report"]),
    )

    return {
        "trial_log": trial_log,
        "runs_used": state.get("runs_used", 0) + 1,
        "best_trial": best,
        "current_experiment": None,
        "last_error": None,
    }


def evaluate_stop(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Décide de continuer ou d'arrêter, selon le `stop_mode`.

    Le budget (`max_runs`) est un garde-fou dur appliqué dans les deux modes. En
    mode « convergence » l'arrêt est déterministe (cible + convergence) ; en mode
    « agent » la décision (hors budget) est déléguée au LLM.
    """
    # Garde-fou dur commun aux deux modes : budget épuisé
    if state.get("runs_used", 0) >= state["max_runs"]:
        logger.info("stop | budget épuisé (%d runs)", state["max_runs"])
        return {"decision": "stop", "stop_reason": "budget épuisé"}

    if state.get("stop_mode", "convergence") == "agent":
        return _evaluate_stop_agent(state, llm)

    return _evaluate_stop_convergence(state)


def _evaluate_stop_convergence(state: AgentState) -> dict:
    """Critères d'arrêt déterministes : seuil cible puis convergence."""
    trial_log = state.get("trial_log", [])
    best = state.get("best_trial")

    # Seuil cible atteint
    if best is not None and best["eval_f1"] >= state["target_f1"]:
        logger.info("stop | seuil cible atteint (eval_f1=%.4f)", best["eval_f1"])
        return {"decision": "stop", "stop_reason": "seuil cible atteint"}

    # Convergence : pas d'amélioration significative sur les K derniers essais
    patience = state["patience"]
    evals = [t["eval_f1"] for t in trial_log]
    if len(evals) > patience:
        recent_best = max(evals[-patience:])
        prior_best = max(evals[:-patience])
        if recent_best - prior_best < state["epsilon"]:
            logger.info("stop | convergence (gain < %g sur %d essais)",
                        state["epsilon"], patience)
            return {
                "decision": "stop",
                "stop_reason": f"convergence (gain < {state['epsilon']:g} sur {patience} essais)",
            }

    return {"decision": "continue", "stop_reason": None}


def _evaluate_stop_agent(state: AgentState, llm: BaseChatModel | None) -> dict:
    """Délègue la décision d'arrêt au LLM (budget déjà borné en amont)."""
    if llm is None:
        raise ValueError("Le mode d'arrêt 'agent' requiert un LLM (llm=None).")

    structured = llm.with_structured_output(StopDecision)
    messages = [
        SystemMessage(content=STOP_SYSTEM_PROMPT),
        HumanMessage(content=build_stop_context(state)),
    ]
    result: StopDecision = structured.invoke(messages)
    logger.info("evaluate (agent) | %s — %s", result.decision, result.reason)
    return {"decision": result.decision, "stop_reason": result.reason}


def route_after_eval(state: AgentState) -> str:
    """Arête conditionnelle : boucler ou terminer."""
    return state.get("decision", "continue")
