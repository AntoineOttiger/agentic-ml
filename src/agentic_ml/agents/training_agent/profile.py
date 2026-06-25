"""Résolution du run de données et construction du profil de dataset.

Le profil est le contexte statique injecté dès l'initialisation de l'agent
(cf. `agentic-archi.md` § « Contexte injecté dès l'initialisation »).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_ml.config import PREP_DATA_DIR, TARGET_COL, TRAIN_FILE
from agentic_ml.training.dataset import resolve_run_dir


def latest_prepared_run() -> str:
    """Renvoie l'identifiant du run le plus récent de `data/02_prepared/`.

    S'appuie sur la convention de nommage de `DataSplitter` : `<dataset>_NNN`,
    le plus grand suffixe numérique étant le plus récent.
    """
    candidates: list[tuple[int, str]] = []
    if PREP_DATA_DIR.is_dir():
        for p in PREP_DATA_DIR.iterdir():
            if not p.is_dir() or "_" not in p.name:
                continue
            suffix = p.name.rsplit("_", 1)[1]
            if suffix.isdigit():
                candidates.append((int(suffix), p.name))

    if not candidates:
        raise FileNotFoundError(
            f"Aucun run préparé dans {PREP_DATA_DIR}. "
            f"Lancez d'abord `python scripts/prepare_data.py`."
        )
    return max(candidates)[1]


def build_dataset_profile(run: str | Path) -> dict[str, Any]:
    """Construit le profil complet d'un run en un seul appel.

    Combine `metadata.json` (distribution des classes, tailles) et `train.csv`
    (ranges des features) pour fournir taille, features, classes et type de tâche.
    """
    run_dir = resolve_run_dir(run)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    train_df = pd.read_csv(run_dir / TRAIN_FILE)

    feature_columns = metadata.get(
        "feature_columns", [c for c in train_df.columns if c != TARGET_COL]
    )
    feature_ranges = {
        col: {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
        }
        for col in feature_columns
    }

    train_info = metadata.get("splits", {}).get("train", {})
    class_distribution = train_info.get(
        "class_distribution",
        train_df[TARGET_COL].value_counts().to_dict(),
    )

    return {
        "run_id": metadata.get("run_id", run_dir.name),
        "task": "classification",
        "target_column": metadata.get("target_column", TARGET_COL),
        "n_train_rows": int(train_info.get("n_rows", len(train_df))),
        "feature_columns": feature_columns,
        "feature_ranges": feature_ranges,
        "classes": list(class_distribution.keys()),
        "class_distribution": class_distribution,
    }
