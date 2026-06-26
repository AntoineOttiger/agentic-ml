"""Chargement d'un run de données préparées pour l'entraînement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from agentic_ml.config import PREP_DATA_DIR, TARGET_COL, TRAIN_FILE, VAL_FILE


@dataclass
class PreparedData:
    """Données d'un run, prêtes pour fit/predict."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    label_encoder: LabelEncoder
    feature_columns: list[str]


def resolve_run_dir(run: str | Path) -> Path:
    """Résout un identifiant de run ('iris_001_001') ou un chemin complet en dossier existant."""
    run_dir = Path(run)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = PREP_DATA_DIR / str(run)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Dossier de run introuvable : {run_dir}")

    for fname in (TRAIN_FILE, VAL_FILE):
        if not (run_dir / fname).is_file():
            raise FileNotFoundError(f"Fichier '{fname}' manquant dans {run_dir}")

    return run_dir


def load_prepared_run(run: str | Path) -> PreparedData:
    """Charge train/val d'un run, sépare X/y et encode la cible en entiers.

    L'encodage des labels est requis par XGBoost et neutre pour RF/SVM ; on
    l'applique uniformément pour garder un contrat de données unique.
    """
    run_dir = resolve_run_dir(run)

    train_df = pd.read_csv(run_dir / TRAIN_FILE)
    val_df = pd.read_csv(run_dir / VAL_FILE)

    feature_columns = [c for c in train_df.columns if c != TARGET_COL]

    encoder = LabelEncoder().fit(train_df[TARGET_COL])

    return PreparedData(
        X_train=train_df[feature_columns],
        y_train=encoder.transform(train_df[TARGET_COL]),
        X_val=val_df[feature_columns],
        y_val=encoder.transform(val_df[TARGET_COL]),
        label_encoder=encoder,
        feature_columns=feature_columns,
    )
