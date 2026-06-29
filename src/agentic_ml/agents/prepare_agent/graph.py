"""Assemblage du StateGraph de préparation et point d'entrée `run_prepare_agent`.

Flux : load_input → (cleaning_step)* → (fe_step)* → split → finalize.
Les deux phases agentiques sont des boucles conditionnelles : on y reste tant que
l'agent n'a pas déclaré « finish » et que les garde-fous de budget tiennent.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agentic_ml.config import (
    AGENT_MODEL,
    DEFAULT_RAW_FILE,
    MAX_PHASE_ATTEMPTS,
    TARGET_COL,
)
from agentic_ml.data_manager.prepare_data import next_run_folder

from agentic_ml.agents.prepare_agent.nodes import (
    cleaning_step,
    fe_step,
    finalize,
    load_input,
    make_llm,
    route_cleaning,
    route_fe,
    split_node,
)
from agentic_ml.agents.prepare_agent.state import PrepConfig, PrepState


def build_prepare_graph(llm: BaseChatModel):
    """Construit et compile le graphe du pipeline de préparation."""
    graph = StateGraph(PrepState)

    graph.add_node("load_input", load_input)
    graph.add_node("cleaning_step", partial(cleaning_step, llm=llm))
    graph.add_node("fe_step", partial(fe_step, llm=llm))
    graph.add_node("split_node", split_node)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_input")
    graph.add_edge("load_input", "cleaning_step")
    graph.add_conditional_edges(
        "cleaning_step",
        route_cleaning,
        {"cleaning": "cleaning_step", "feature_engineering": "fe_step"},
    )
    graph.add_conditional_edges(
        "fe_step",
        route_fe,
        {"feature_engineering": "fe_step", "split": "split_node"},
    )
    graph.add_edge("split_node", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_prepare_agent(
    *,
    input_csv: str | Path = DEFAULT_RAW_FILE,
    target_column: str = TARGET_COL,
    output_dir: str | Path | None = None,
    task_type: str = "classification",
    seed: int = 42,
    dry_run: bool = False,
    enable_permutation: bool = False,
    model: str = AGENT_MODEL,
) -> PrepState:
    """Exécute le pipeline cleaning → FE → split de bout en bout.

    `input_csv` et `target_column` défaulttent aux constantes de `config.py`.
    `output_dir` n'est pas censé être fourni : il est dérivé automatiquement en
    `data/01_prepared/<dataset>_<NNN>` (prochain index versionné), aligné sur la
    convention de `DataSplitter`. Renvoie l'état final (`PrepState`), dont
    `output_path` pointe vers les artefacts.
    """
    input_csv = Path(input_csv)
    if output_dir is None:
        output_dir = next_run_folder(input_csv)
    config = PrepConfig(
        input_csv=str(input_csv.as_posix()),
        target_column=target_column,
        task_type=task_type,
        output_dir=str(Path(output_dir).as_posix()),
        seed=seed,
        dry_run=dry_run,
        enable_permutation=enable_permutation,
    )
    initial_state = PrepState(config=config)

    app = build_prepare_graph(make_llm(model))
    # Plafond de sécurité : deux phases à MAX_PHASE_ATTEMPTS itérations + nœuds fixes.
    recursion_limit = MAX_PHASE_ATTEMPTS * 2 + 12
    final = app.invoke(initial_state, {"recursion_limit": recursion_limit})
    # LangGraph renvoie un dict ou un modèle selon la version : on normalise.
    return final if isinstance(final, PrepState) else PrepState(**final)
