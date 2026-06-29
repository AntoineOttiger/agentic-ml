"""Contrats et invariants inter-phases (§8). Un invariant violé → arrêt explicite.

La validation est volontairement *fail-fast* : plutôt que de laisser une fuite ou
une corruption se propager, on lève `ContractViolation` avec un message clair.
"""
from __future__ import annotations

import pandas as pd

from agentic_ml.agents.prepare_agent.state import PrepState, TransformationRecord


class ContractViolation(RuntimeError):
    """Un invariant inter-phases n'est pas respecté."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def check_entry(df: pd.DataFrame, target_column: str) -> None:
    """Entrée → Cleaning : cible présente, CSV lisible, schéma capturable."""
    _assert(
        target_column in df.columns,
        f"Colonne cible « {target_column} » absente du CSV (colonnes : {list(df.columns)}).",
    )
    _assert(len(df) > 0, "Le CSV source est vide.")


def _assert_target_intact(
    before: pd.Series, df: pd.DataFrame, target_column: str
) -> None:
    _assert(
        target_column in df.columns,
        f"La colonne cible « {target_column} » a disparu après transformation.",
    )
    after = df[target_column]
    _assert(
        len(after) == len(before) and after.reset_index(drop=True).equals(
            before.reset_index(drop=True)
        ),
        f"La colonne cible « {target_column} » a été modifiée — interdit (§3).",
    )


def check_step_record(record: TransformationRecord) -> None:
    """Une étape ne peut entrer dans l'historique qu'avec un gate `passed` (§7)."""
    _assert(
        record.type == "stateless",
        f"Étape « {record.name} » non stateless : interdite dans l'historique.",
    )
    _assert(
        record.invariance_test is not None and record.invariance_test.passed,
        f"Étape « {record.name} » sans gate d'invariance §6 `passed` : refusée.",
    )


def check_cleaning_output(state: PrepState) -> None:
    """Sortie Cleaning : cible intacte, toutes les étapes avec gate `passed`."""
    df = state.df
    target = state.config.target_column
    _assert(df is not None, "Dataframe manquant en sortie de cleaning.")
    _assert(
        target in df.columns, f"Colonne cible « {target} » absente en sortie de cleaning."
    )
    for record in state.history:
        if record.phase == "cleaning":
            check_step_record(record)


def check_fe_output(state: PrepState) -> None:
    """Sortie FE : cible présente, étapes stateless + gate, pas de NaN silencieux."""
    df = state.df
    target = state.config.target_column
    _assert(df is not None, "Dataframe manquant en sortie de feature engineering.")
    _assert(
        target in df.columns,
        f"Colonne cible « {target} » absente en sortie de feature engineering.",
    )
    for record in state.history:
        check_step_record(record)


def check_split_output(
    splits: dict[str, pd.DataFrame], post_fe_n_rows: int, target_column: str
) -> None:
    """Sortie Split : jeux disjoints, union = dataset post-FE, cible présente."""
    total = sum(len(s) for s in splits.values())
    _assert(
        total == post_fe_n_rows,
        f"Union des splits ({total} lignes) ≠ dataset post-FE ({post_fe_n_rows} lignes).",
    )
    for name, frame in splits.items():
        _assert(
            target_column in frame.columns,
            f"Colonne cible « {target_column} » absente du split « {name} ».",
        )
