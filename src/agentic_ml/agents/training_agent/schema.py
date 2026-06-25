"""Sortie structurée du LLM pour une proposition d'essai.

Le LLM renvoie une liste de `HyperparameterRange` (forme robuste pour la sortie
structurée) que `to_search_space` convertit au format dict attendu par
`agentic_ml.training.search_space`.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ModelType = Literal[
    "xgboost",
    "random_forest",
    "svm",
    "logistic_regression",
    "k_nearest_neighbors",
    "gaussian_nb",
    "mlp",
    "lda",
    "extra_trees",
    "hist_gradient_boosting",
]


class HyperparameterRange(BaseModel):
    """Plage d'un hyperparamètre à explorer."""

    name: str = Field(description="Nom exact de l'hyperparamètre (cf. schéma du modèle).")
    type: Literal["int", "float", "categorical"]
    low: Optional[float] = Field(default=None, description="Borne basse (int/float).")
    high: Optional[float] = Field(default=None, description="Borne haute (int/float).")
    log: bool = Field(default=False, description="Échelle logarithmique (int/float).")
    step: Optional[float] = Field(default=None, description="Pas (int/float, optionnel).")
    choices: Optional[list[str]] = Field(
        default=None, description="Valeurs possibles (categorical)."
    )


class Experiment(BaseModel):
    """Proposition d'essai : hypothèse explicite + configuration."""

    hypothesis: str = Field(
        description="Justification écrite de la configuration, avant de lancer le run."
    )
    model_type: ModelType
    search_space: list[HyperparameterRange] = Field(
        description="Plages d'hyperparamètres à explorer pour ce modèle."
    )


class StopDecision(BaseModel):
    """Décision d'arrêt prise par le LLM (mode d'arrêt « agent »)."""

    decision: Literal["continue", "stop"]
    reason: str = Field(
        description="Justification courte de la décision de poursuivre ou d'arrêter."
    )


def to_search_space(ranges: list[HyperparameterRange]) -> dict[str, dict[str, Any]]:
    """Convertit la liste de plages en dict au format `search_space`."""
    space: dict[str, dict[str, Any]] = {}
    for r in ranges:
        if r.type in ("int", "float"):
            spec: dict[str, Any] = {"type": r.type, "low": r.low, "high": r.high}
            if r.log:
                spec["log"] = True
            if r.step is not None:
                spec["step"] = int(r.step) if r.type == "int" else r.step
        else:  # categorical
            spec = {"type": "categorical", "choices": r.choices or []}
        space[r.name] = spec
    return space
