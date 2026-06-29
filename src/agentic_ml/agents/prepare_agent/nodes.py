"""Nœuds du pipeline de préparation (LangGraph).

Flux : load_input → [cleaning_step]* → [fe_step]* → split → finalize.

Chaque pas d'une phase (cleaning/FE) suit le contrat §5 : l'agent décide, le code
exécute. Une transformation n'est appliquée et enregistrée QUE si elle passe le
gate d'invariance §6 — sinon le diagnostic est renvoyé à l'agent (§6.4).
"""
from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import pandas as pd

from agentic_ml.config import (
    AGENT_MODEL,
    AGENT_PROVIDER,
    MAX_CLEANING_STEPS,
    MAX_FE_STEPS,
    MAX_PHASE_ATTEMPTS,
)
from agentic_ml.data_manager.prepare_data import DataSplitter
from agentic_ml.utils.rate_limiter import RateLimitCallback, get_rate_limiter

from agentic_ml.agents.prepare_agent import contracts
from agentic_ml.agents.prepare_agent.artifacts import write_artifacts
from agentic_ml.agents.prepare_agent.executor import (
    ExecutorError,
    apply_transform,
    compile_transform,
)
from agentic_ml.agents.prepare_agent.invariance import run_invariance_gate
from agentic_ml.agents.prepare_agent.prompts import (
    build_step_context,
    cleaning_system_prompt,
    fe_system_prompt,
)
from agentic_ml.agents.prepare_agent.recipe import script_path_for
from agentic_ml.agents.prepare_agent.schema import StepProposal
from agentic_ml.agents.prepare_agent.state import (
    PrepState,
    TransformationRecord,
)
from agentic_ml.agents.prepare_agent.tools import (
    capture_schema,
    hash_file,
    hash_frame,
    profile_dataframe,
)

logger = logging.getLogger("agentic_ml.agents.prepare_agent")


def make_llm(model: str = AGENT_MODEL, *, temperature: float = 0.3) -> BaseChatModel:
    """Instancie le client LLM selon AGENT_PROVIDER, avec rate limiting."""
    callback = RateLimitCallback(get_rate_limiter())
    if AGENT_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, callbacks=[callback])
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(model=model, temperature=temperature, callbacks=[callback])


# --------------------------------------------------------------------------- #
# Phase 0 — chargement
# --------------------------------------------------------------------------- #
def load_input(state: PrepState) -> dict:
    """Lit le CSV, isole la cible, capture schéma + provenance (§8 entrée)."""
    cfg = state.config
    df = pd.read_csv(cfg.input_csv)
    contracts.check_entry(df, cfg.target_column)
    logger.info("load | %s : %d lignes × %d colonnes", cfg.input_csv, len(df), df.shape[1])
    return {
        "df": df,
        "source_hash": hash_file(cfg.input_csv),
        "current_schema": capture_schema(df),
        "phase": "cleaning",
    }


# --------------------------------------------------------------------------- #
# Cœur partagé d'un pas agentique (cleaning ou FE)
# --------------------------------------------------------------------------- #
def _agent_step(
    state: PrepState,
    llm: BaseChatModel,
    *,
    phase: str,
    system_prompt: str,
    steps_used: int,
    max_steps: int,
) -> dict:
    """Un pas : l'agent propose → gate §6 → application + enregistrement, ou rejet.

    Renvoie un dict d'updates partiels de `PrepState`. Ne fait jamais progresser
    le pipeline si la transformation échoue au gate.
    """
    cfg = state.config
    df = state.df
    profile = profile_dataframe(df, cfg.target_column)
    context = build_step_context(
        phase=phase,
        profile=profile,
        history=state.history,
        steps_used=steps_used,
        max_steps=max_steps,
        last_error=state.last_error,
    )
    structured = llm.with_structured_output(StepProposal)
    proposal: StepProposal = structured.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=context)]
    )

    finished_field = "finished_cleaning" if phase == "cleaning" else "finished_fe"
    attempts_field = "cleaning_attempts" if phase == "cleaning" else "fe_attempts"
    steps_field = "cleaning_steps" if phase == "cleaning" else "fe_steps"
    attempts = getattr(state, attempts_field) + 1

    if proposal.action != "transform" or not proposal.code:
        logger.info("%s | l'agent termine la phase : %s", phase, proposal.rationale)
        return {finished_field: True, attempts_field: attempts, "last_error": None}

    name = proposal.name or f"step_{steps_used + 1}"

    # 1. Compilation du script de l'agent.
    try:
        fn = compile_transform(proposal.code)
    except ExecutorError as exc:
        logger.warning("%s | script rejeté (compilation) : %s", phase, exc)
        return {attempts_field: attempts, "last_error": str(exc)}

    # 2. Gate d'invariance par ligne (§6) — AVANT toute application.
    try:
        report = run_invariance_gate(fn, df, enable_permutation=cfg.enable_permutation)
    except ExecutorError as exc:
        logger.warning("%s | script rejeté (exécution gate) : %s", phase, exc)
        return {attempts_field: attempts, "last_error": str(exc)}

    if not report.passed:
        logger.warning("%s | gate ÉCHEC « %s » : %s", phase, name, report.diagnostic)
        return {
            attempts_field: attempts,
            "last_error": (
                f"Transformation « {name} » rejetée par le gate d'invariance §6. "
                f"{report.diagnostic} Propose une alternative row-local ou « finish »."
            ),
        }

    # 3. Application déterministe + protection de la cible (§3).
    target_before = df[cfg.target_column]
    try:
        new_df = apply_transform(fn, df)
    except ExecutorError as exc:
        return {attempts_field: attempts, "last_error": str(exc)}

    try:
        contracts._assert_target_intact(target_before, new_df, cfg.target_column)
    except contracts.ContractViolation as exc:
        logger.warning("%s | transformation rejetée (cible touchée) : %s", phase, exc)
        return {attempts_field: attempts, "last_error": str(exc)}

    # 4. Enregistrement dans l'historique (recette §7).
    input_schema = capture_schema(df)
    output_schema = capture_schema(new_df)
    record = TransformationRecord(
        phase=phase,
        name=name,
        description=proposal.description or name,
        type="stateless",
        params={},
        script_path=script_path_for(len(state.history) + 1, phase, name),
        code=proposal.code,
        invariance_test=report,
        input_schema=input_schema,
        output_schema=output_schema,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        input_hash=hash_frame(df),
        output_hash=hash_frame(new_df),
    )
    contracts.check_step_record(record)

    logger.info(
        "%s | appliqué « %s » (%d→%d colonnes) | gate passed",
        phase, name, df.shape[1], new_df.shape[1],
    )
    return {
        "df": new_df,
        "current_schema": output_schema,
        "history": state.history + [record],
        steps_field: steps_used + 1,
        attempts_field: attempts,
        "last_error": None,
    }


def cleaning_step(state: PrepState, llm: BaseChatModel) -> dict:
    return _agent_step(
        state, llm,
        phase="cleaning",
        system_prompt=cleaning_system_prompt(state.config.target_column),
        steps_used=state.cleaning_steps,
        max_steps=MAX_CLEANING_STEPS,
    )


def fe_step(state: PrepState, llm: BaseChatModel) -> dict:
    return _agent_step(
        state, llm,
        phase="feature_engineering",
        system_prompt=fe_system_prompt(state.config.target_column),
        steps_used=state.fe_steps,
        max_steps=MAX_FE_STEPS,
    )


# --------------------------------------------------------------------------- #
# Routage des boucles de phase
# --------------------------------------------------------------------------- #
def route_cleaning(state: PrepState) -> str:
    """Boucle cleaning tant que l'agent n'a pas fini et que le budget tient."""
    if (
        state.finished_cleaning
        or state.cleaning_steps >= MAX_CLEANING_STEPS
        or state.cleaning_attempts >= MAX_PHASE_ATTEMPTS
    ):
        contracts.check_cleaning_output(state)
        logger.info("cleaning | phase terminée (%d transformations)", state.cleaning_steps)
        return "feature_engineering"
    return "cleaning"


def route_fe(state: PrepState) -> str:
    """Boucle FE tant que l'agent n'a pas fini et que le budget tient."""
    if (
        state.finished_fe
        or state.fe_steps >= MAX_FE_STEPS
        or state.fe_attempts >= MAX_PHASE_ATTEMPTS
    ):
        contracts.check_fe_output(state)
        logger.info("fe | phase terminée (%d features)", state.fe_steps)
        return "split"
    return "feature_engineering"


# --------------------------------------------------------------------------- #
# Phase 3 — split + finalisation
# --------------------------------------------------------------------------- #
def split_node(state: PrepState) -> dict:
    """Partitionne (déterministe, §9) puis valide le contrat de sortie (§8)."""
    cfg = state.config
    if cfg.dry_run:
        logger.info("split | dry-run : partitions non matérialisées")
        return {"phase": "split"}

    splitter = DataSplitter(source_file=cfg.input_csv, random_seed=cfg.seed)
    splits = splitter.split_frame(state.df, target_column=cfg.target_column)
    contracts.check_split_output(splits, len(state.df), cfg.target_column)
    logger.info(
        "split | %s", {name: len(frame) for name, frame in splits.items()}
    )
    return {"splits": splits, "phase": "split"}


def finalize(state: PrepState) -> dict:
    """Écrit les artefacts (§10) et clôt le pipeline."""
    out = write_artifacts(state)
    logger.info("finalize | artefacts écrits dans %s", out)
    return {"output_path": str(out), "phase": "done"}
