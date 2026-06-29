"""Sortie structurée du LLM pour une étape de cleaning ou de feature engineering.

À chaque tour, l'agent **décide** (§5) : soit il propose UNE transformation (un
script Python autonome `transform(df) -> df`), soit il déclare la phase terminée.
Le code émis n'a aucune valeur de garantie tant qu'il n'a pas passé le gate (§6).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StepProposal(BaseModel):
    """Décision de l'agent pour le prochain pas d'une phase."""

    action: str = Field(
        description=(
            "« transform » pour appliquer une nouvelle transformation, "
            "« finish » lorsque la phase est terminée."
        ),
    )
    rationale: str = Field(
        description="Justification de la décision (pourquoi cette transformation, ou pourquoi finir)."
    )
    name: Optional[str] = Field(
        default=None,
        description="Nom court en snake_case de la transformation (requis si action=transform).",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description courte de l'effet de la transformation (si action=transform).",
    )
    code: Optional[str] = Field(
        default=None,
        description=(
            "Code source Python COMPLET définissant `def transform(df: pd.DataFrame) "
            "-> pd.DataFrame:` (requis si action=transform). `pd` et `np` sont fournis. "
            "La fonction doit être row-local, déterministe, ne pas réinitialiser l'index, "
            "et ne jamais modifier la colonne cible."
        ),
    )
