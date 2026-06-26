"""Système agentique de preprocessing / feature engineering (LangGraph + Mistral).

Indépendant et isolé du `training_agent` : ne partage ni state, ni graphe, ni
agents. Opère directement sur un DataFrame en mémoire (data/00_raw → data/01_preproc).
"""
from agentic_ml.agents.preproc_agent.graph import build_preproc_graph, run_preproc_agent
from agentic_ml.agents.preproc_agent.state import MLPipelineState

__all__ = ["run_preproc_agent", "build_preproc_graph", "MLPipelineState"]
