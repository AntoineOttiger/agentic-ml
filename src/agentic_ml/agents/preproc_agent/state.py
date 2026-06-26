"""État partagé du système agentique de preprocessing / feature engineering.

Réinjecté à chaque tour de la boucle LangGraph (Analyse → Preprocessing →
Feature Engineering → Analyse). L'historique des transformations
(`applied_transformations`, `created_features`) ne doit jamais être réinitialisé
entre itérations : il sert à éviter les doublons.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

import pandas as pd


class MLPipelineState(TypedDict, total=False):
    """État de la boucle de preprocessing.

    `total=False` : les champs sont peuplés progressivement par les nœuds.
    """

    # Données
    dataframe: pd.DataFrame            # DataFrame courant (transformé en place)
    target_column: str                 # Nom de la colonne cible

    # Historique des transformations (anti-doublon)
    applied_transformations: list[str]  # ex: ["impute_median(SepalWidthCm)", "drop(Id)"]
    created_features: list[str]         # ex: ["petal_ratio", "sepal_area"]

    # Communication entre agents
    analysis_report: dict[str, Any]    # Rapport de l'Agent Analyse (sérialisable)
    actions_to_apply: list[dict[str, Any]]  # Actions recommandées restantes
    generated_code: str                # Dernier code Python généré
    execution_error: Optional[str]     # Erreur d'exécution si applicable

    # Contrôle de boucle
    iteration_count: int
    max_iterations: int                # Limite de sécurité (défaut: config.DEFAULT_MAX_ITERATIONS)
    should_continue: bool

    # Routage (écrit par le nœud analyse, lu par le routeur déterministe)
    route: Literal["preprocessing", "feature_engineering", "end"]
