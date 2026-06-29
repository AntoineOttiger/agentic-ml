"""État partagé du pipeline de préparation (Pydantic v2) et modèles de provenance.

Un unique objet `PrepState` circule à travers les trois phases (cleaning →
feature engineering → split). Le dataframe courant est tenu **en mémoire** (§4 :
« référence vers le dataframe courant ») ; il est exclu de toute sérialisation.

Les modèles `TransformationRecord` / `InvarianceReport` constituent la **recette
rejouable** (§7) : chaque étape n'y figure que si son gate d'invariance (§6) est
`passed`.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Provenance / recette (§7)
# --------------------------------------------------------------------------- #
class ColumnInfo(BaseModel):
    """Une colonne du schéma : nom + dtype pandas (str)."""

    name: str
    dtype: str


class InvarianceCheck(BaseModel):
    """Résultat d'une vérification élémentaire du gate (§6.2)."""

    kind: Literal["determinism", "subsample", "permutation"]
    fraction: Optional[float] = None
    seed: Optional[int] = None
    passed: bool


class InvarianceReport(BaseModel):
    """Synthèse du gate d'invariance par ligne (§6) pour une transformation."""

    passed: bool
    sample_size: int
    checks: list[InvarianceCheck] = Field(default_factory=list)
    diagnostic: Optional[str] = None  # exemple de ligne divergente en cas d'échec


class TransformationRecord(BaseModel):
    """Étape rejouable de la recette (§7). Sans gate `passed`, n'existe pas."""

    phase: Literal["cleaning", "feature_engineering"]
    name: str
    description: str
    type: Literal["stateless"] = "stateless"
    params: dict[str, Any] = Field(default_factory=dict)
    script_path: str  # chemin relatif du script versionné (transforms/*.py)
    code: str  # code source intégral du script invoqué
    invariance_test: InvarianceReport
    input_schema: list[ColumnInfo]
    output_schema: list[ColumnInfo]
    timestamp: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None


# --------------------------------------------------------------------------- #
# Configuration d'entrée (§3)
# --------------------------------------------------------------------------- #
class PrepConfig(BaseModel):
    """Configuration explicite du pipeline — on n'infère rien silencieusement (§3)."""

    input_csv: str
    target_column: str
    task_type: Literal["classification"]
    output_dir: str
    seed: int = 42
    dry_run: bool = False
    enable_permutation: bool = False


# --------------------------------------------------------------------------- #
# État LangGraph
# --------------------------------------------------------------------------- #
class PrepState(BaseModel):
    """État muté de façon contrôlée par chaque phase.

    `arbitrary_types_allowed` autorise le portage du `DataFrame` courant entre
    nœuds. `df` et `splits` sont exclus de toute sérialisation JSON.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: PrepConfig

    # Référence vers le dataframe courant (en mémoire) + provenance
    df: Optional[pd.DataFrame] = Field(default=None, exclude=True, repr=False)
    source_hash: Optional[str] = None
    current_schema: list[ColumnInfo] = Field(default_factory=list)

    # Historique des transformations (recette §7)
    history: list[TransformationRecord] = Field(default_factory=list)

    # Pilotage des boucles de phase
    phase: Literal["load", "cleaning", "feature_engineering", "split", "done"] = "load"
    cleaning_steps: int = 0
    fe_steps: int = 0
    cleaning_attempts: int = 0
    fe_attempts: int = 0
    finished_cleaning: bool = False
    finished_fe: bool = False

    # Dernier diagnostic du gate, renvoyé à l'agent pour qu'il corrige (§6.4)
    last_error: Optional[str] = None

    # Résultats de la phase 3 (frames en mémoire) + chemin des artefacts
    splits: Optional[dict[str, pd.DataFrame]] = Field(
        default=None, exclude=True, repr=False
    )
    output_path: Optional[str] = None
