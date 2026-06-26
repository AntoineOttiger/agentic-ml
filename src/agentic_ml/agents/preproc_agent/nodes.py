"""Nœuds LangGraph du système preproc : Analyse, Preprocessing, Feature Engineering.

Chaque nœud est synchrone, reçoit le `state` + un `llm` injecté par `partial`,
et renvoie un dict mergé dans le state. Le routeur `route_after_analysis` est
déterministe (pas d'appel LLM) et lit la clé `route` posée par l'Agent Analyse.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

from agentic_ml.agents.preproc_agent.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    FEATURE_ENGINEERING_SYSTEM_PROMPT,
    PREPROCESSING_SYSTEM_PROMPT,
    build_analysis_context,
    build_fe_context,
    build_preproc_context,
    build_retry_suffix,
)
from agentic_ml.agents.preproc_agent.sandbox import execute_code
from agentic_ml.agents.preproc_agent.schema import AnalysisReport, GeneratedCode
from agentic_ml.agents.preproc_agent.state import MLPipelineState
from agentic_ml.config import MAX_CODE_RETRIES, PREPROC_AGENT_MODEL

logger = logging.getLogger("agentic_ml.agents.preproc_agent")


def make_llm(model: str = PREPROC_AGENT_MODEL, *, temperature: float = 0.2) -> ChatMistralAI:
    """Instancie le client Mistral (clé lue dans MISTRAL_API_KEY)."""
    return ChatMistralAI(model=model, temperature=temperature)


def _action_signature(action: dict[str, Any]) -> str:
    """Signature compacte d'une action, pour comparer à l'historique."""
    cols = ",".join(action.get("columns", []))
    return f"{action.get('action', '?')}({cols})"


def agent_analyse(state: MLPipelineState, *, llm: ChatMistralAI) -> dict[str, Any]:
    """Profile les données, produit le rapport et décide du routage."""
    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=build_analysis_context(state)),
    ]
    report: AnalysisReport = llm.with_structured_output(AnalysisReport).invoke(messages)

    iteration = state.get("iteration_count", 0) + 1
    history = set(state.get("applied_transformations", [])) | set(
        state.get("created_features", [])
    )

    # Filtrer les actions déjà réalisées (anti-doublon).
    actions = [
        a.model_dump()
        for a in report.recommended_actions
        if _action_signature(a.model_dump()) not in history
    ]

    has_preproc = any(a["type"] == "preprocessing" for a in actions)
    has_fe = any(a["type"] == "feature_engineering" for a in actions)
    should_continue = not report.is_clean and bool(actions)

    if not should_continue:
        route = "end"
    elif has_preproc:
        route = "preprocessing"
    elif has_fe:
        route = "feature_engineering"
    else:
        route = "end"

    logger.info(
        "analyse | itération=%d | is_clean=%s | actions=%d | route=%s",
        iteration,
        report.is_clean,
        len(actions),
        route,
    )

    return {
        "analysis_report": report.model_dump(),
        "actions_to_apply": actions,
        "iteration_count": iteration,
        "should_continue": should_continue,
        "route": route,
    }


def _apply_action(
    state: MLPipelineState,
    *,
    llm: ChatMistralAI,
    system_prompt: str,
    context: str,
) -> tuple[Any, list[str], str, str | None]:
    """Génère puis exécute le code d'une action, avec retry sur erreur.

    Returns:
        ``(df, applied, code, error)`` — `df` inchangé et `error` non nul si tous
        les essais ont échoué.
    """
    structured = llm.with_structured_output(GeneratedCode)
    df = state["dataframe"]
    target = state["target_column"]
    user_content = context
    code = ""
    error: str | None = None

    for attempt in range(1, MAX_CODE_RETRIES + 1):
        result: GeneratedCode = structured.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )
        code = result.code
        # Rendre la cible disponible au code généré sous le nom `target`.
        wrapped = f"target = {target!r}\n{code}"
        new_df, error = execute_code(wrapped, df)
        if error is None:
            return new_df, result.applied, code, None
        logger.warning("exécution | tentative %d/%d échouée : %s", attempt, MAX_CODE_RETRIES, error)
        user_content = context + build_retry_suffix(code, error)

    return df, [], code, error


def agent_preprocessing(state: MLPipelineState, *, llm: ChatMistralAI) -> dict[str, Any]:
    """Applique la première action de preprocessing en attente."""
    actions = [a for a in state.get("actions_to_apply", []) if a["type"] == "preprocessing"]
    if not actions:
        return {}

    action = actions[0]
    df, applied, code, error = _apply_action(
        state,
        llm=llm,
        system_prompt=PREPROCESSING_SYSTEM_PROMPT,
        context=build_preproc_context(state, action),
    )

    if error is not None:
        logger.warning("preprocessing | action abandonnée : %s", _action_signature(action))
        return {"generated_code": code, "execution_error": error}

    labels = applied or [_action_signature(action)]
    logger.info("preprocessing | appliqué : %s", ", ".join(labels))
    return {
        "dataframe": df,
        "applied_transformations": state.get("applied_transformations", []) + labels,
        "generated_code": code,
        "execution_error": None,
    }


def agent_feature_engineering(state: MLPipelineState, *, llm: ChatMistralAI) -> dict[str, Any]:
    """Applique la première action de feature engineering en attente."""
    actions = [
        a for a in state.get("actions_to_apply", []) if a["type"] == "feature_engineering"
    ]
    if not actions:
        return {}

    action = actions[0]
    df, applied, code, error = _apply_action(
        state,
        llm=llm,
        system_prompt=FEATURE_ENGINEERING_SYSTEM_PROMPT,
        context=build_fe_context(state, action),
    )

    if error is not None:
        logger.warning("feature_engineering | action abandonnée : %s", _action_signature(action))
        return {"generated_code": code, "execution_error": error}

    labels = applied or [_action_signature(action)]
    logger.info("feature_engineering | créé : %s", ", ".join(labels))
    return {
        "dataframe": df,
        "created_features": state.get("created_features", []) + labels,
        "generated_code": code,
        "execution_error": None,
    }


def route_after_analysis(state: MLPipelineState) -> str:
    """Routeur déterministe : applique les garde-fous puis suit `state['route']`."""
    if not state.get("should_continue", False):
        return "end"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 0):
        logger.info("routeur | max_iterations atteint (%d)", state.get("max_iterations", 0))
        return "end"
    return state.get("route", "end")
