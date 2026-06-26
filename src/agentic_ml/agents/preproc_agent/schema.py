"""Modèles Pydantic pour les sorties structurées des agents preproc.

- `AnalysisReport` : rapport produit par l'Agent Analyse (structure de la spec).
- `GeneratedCode` : code Python + libellés des transformations, produit par les
  agents Preprocessing et Feature Engineering.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ColumnReport(BaseModel):
    """Diagnostic d'une colonne."""

    dtype: str = Field(description="Type de données (numérique, catégorielle, datetime, texte).")
    null_pct: float = Field(description="Pourcentage de valeurs manquantes.")
    skewness: Optional[float] = Field(default=None, description="Asymétrie (numériques).")
    n_unique: int = Field(description="Cardinalité (nombre de valeurs distinctes).")
    has_outliers: bool = Field(default=False, description="Présence d'outliers (IQR).")
    issues: list[str] = Field(
        default_factory=list,
        description="ex: ['high_nulls', 'skewed', 'high_cardinality'].",
    )


class RecommendedAction(BaseModel):
    """Action recommandée par l'Agent Analyse."""

    type: Literal["preprocessing", "feature_engineering"] = Field(
        description="Catégorie d'agent qui doit traiter l'action."
    )
    action: str = Field(
        description="ex: 'impute_median', 'log_transform', 'create_ratio'."
    )
    columns: list[str] = Field(
        default_factory=list, description="Colonnes concernées par l'action."
    )
    reason: str = Field(description="Justification courte de la recommandation.")


class AnalysisReport(BaseModel):
    """Rapport structuré de l'Agent Analyse (structure JSON de la spec)."""

    columns: dict[str, ColumnReport] = Field(
        default_factory=dict, description="Diagnostic par colonne."
    )
    global_issues: list[str] = Field(
        default_factory=list, description="Problèmes au niveau du dataset."
    )
    recommended_actions: list[RecommendedAction] = Field(
        default_factory=list,
        description="Actions à appliquer ; vide si le dataset est propre.",
    )
    is_clean: bool = Field(
        description="True si aucune action n'est nécessaire (arrêt de la boucle)."
    )


class GeneratedCode(BaseModel):
    """Code Python généré par un agent + libellés des transformations effectuées."""

    code: str = Field(
        description=(
            "Code Python pandas/sklearn valide manipulant le DataFrame `df` "
            "(et `target` si besoin). Aucun import manquant, aucune E/S disque."
        )
    )
    applied: list[str] = Field(
        default_factory=list,
        description=(
            "Libellés des transformations/features réalisées par ce code, "
            "ex: ['impute_median(SepalWidthCm)'] ou ['petal_ratio']."
        ),
    )
