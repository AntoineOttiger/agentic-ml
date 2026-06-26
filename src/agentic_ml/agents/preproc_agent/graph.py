"""Assemblage du StateGraph et point d'entrée `run_preproc_agent`.

Flux : analyse → (preprocessing | feature_engineering | fin), avec retour à
l'analyse après chaque action. La boucle s'arrête quand l'Agent Analyse juge le
dataset propre (`is_clean`) ou quand `max_iterations` est atteint.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from agentic_ml.agents.preproc_agent.nodes import (
    agent_analyse,
    agent_feature_engineering,
    agent_preprocessing,
    make_llm,
    route_after_analysis,
)
from agentic_ml.agents.preproc_agent.results import persist_results
from agentic_ml.agents.preproc_agent.state import MLPipelineState
from agentic_ml.config import (
    DEFAULT_MAX_ITERATIONS,
    PREPROC_AGENT_MODEL,
    PREPROC_DATA_DIR,
    DEFAULT_RAW_FILE,
    TARGET_COL,
)


def build_preproc_graph(llm: BaseChatModel):
    """Construit et compile le graphe de la boucle de preprocessing."""
    graph = StateGraph(MLPipelineState)

    graph.add_node("analyse", partial(agent_analyse, llm=llm))
    graph.add_node("preprocessing", partial(agent_preprocessing, llm=llm))
    graph.add_node("feature_engineering", partial(agent_feature_engineering, llm=llm))

    graph.set_entry_point("analyse")
    graph.add_conditional_edges(
        "analyse",
        route_after_analysis,
        {
            "preprocessing": "preprocessing",
            "feature_engineering": "feature_engineering",
            "end": END,
        },
    )
    graph.add_edge("preprocessing", "analyse")
    graph.add_edge("feature_engineering", "analyse")

    return graph.compile()


def run_preproc_agent(
    *,
    input_csv: Optional[str | Path] = None,
    target_column: str = TARGET_COL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model: str = PREPROC_AGENT_MODEL,
    output_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> MLPipelineState:
    """Exécute la boucle de preprocessing de bout en bout et renvoie l'état final.

    Args:
        input_csv: CSV d'entrée (défaut : `config.DEFAULT_RAW_FILE`).
        target_column: colonne cible à préserver pendant les transformations.
        max_iterations: garde-fou de la boucle de l'orchestrateur.
        model: identifiant du modèle Mistral.
        output_dir: dossier de sortie (défaut : `config.PREPROC_DATA_DIR`).
        dataset_name: préfixe des dossiers versionnés (défaut : nom du CSV).

    Le DataFrame transformé et l'historique sont écrits sur disque à la fin.
    """
    source = Path(input_csv) if input_csv is not None else DEFAULT_RAW_FILE
    df = pd.read_csv(source)

    if target_column not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_column}' introuvable dans {source}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    initial_state: MLPipelineState = {
        "dataframe": df,
        "target_column": target_column,
        "agent_model": model,
        "applied_transformations": [],
        "created_features": [],
        "analysis_report": {},
        "actions_to_apply": [],
        "generated_code": "",
        "execution_error": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "route": "end",
    }

    app = build_preproc_graph(make_llm(model))
    # Plafond de sécurité : au pire une analyse + une action par itération, plus une marge.
    final_state: MLPipelineState = app.invoke(
        initial_state, {"recursion_limit": max_iterations * 3 + 10}
    )

    persist_kwargs: dict = {}
    if output_dir is not None:
        persist_kwargs["output_dir"] = output_dir
    persist_kwargs["dataset_name"] = dataset_name or source.stem.lower()
    persist_results(final_state, **persist_kwargs)

    return final_state
