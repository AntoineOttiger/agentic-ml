"""Fonctions utilitaires : profil du dataframe, capture de schéma, hachage.

Le profil est le contexte factuel injecté à l'agent à chaque tour pour qu'il
décide d'une transformation. Il reste purement descriptif — il ne calcule aucune
statistique qui serait réinjectée comme paramètre dans une transformation (ce qui
introduirait une fuite ; cf. §2).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_ml.agents.prepare_agent.state import ColumnInfo


def capture_schema(df: pd.DataFrame) -> list[ColumnInfo]:
    """Schéma courant : (colonne, dtype) pour chaque colonne."""
    return [ColumnInfo(name=str(c), dtype=str(dt)) for c, dt in df.dtypes.items()]


def _to_python(value: Any) -> Any:
    """Rend une valeur JSON-sérialisable (types numpy → natifs)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def hash_frame(df: pd.DataFrame) -> str:
    """Hash sha256 déterministe du contenu d'un dataframe (audit §7)."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(",".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()


def hash_file(path: str | Path) -> str:
    """Hash sha256 d'un fichier (provenance du CSV source §4)."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def profile_dataframe(
    df: pd.DataFrame, target_column: str, *, max_columns: int = 60, n_examples: int = 3
) -> dict[str, Any]:
    """Profil descriptif du dataframe pour le contexte de l'agent.

    Marque la colonne cible comme protégée et n'en expose que le strict minimum
    (présence, dtype) — jamais sa distribution, qui ne doit influencer aucune
    feature (§3).
    """
    n_rows = len(df)
    columns: list[dict[str, Any]] = []
    for col in list(df.columns)[:max_columns]:
        s = df[col]
        is_target = col == target_column
        info: dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "is_target": is_target,
        }
        if not is_target:
            examples = [
                _to_python(v) for v in s.dropna().unique()[:n_examples]
            ]
            info["examples"] = examples
            info["is_constant"] = bool(s.nunique(dropna=False) <= 1)
        columns.append(info)

    return {
        "n_rows": int(n_rows),
        "n_columns": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "target_column": target_column,
        "columns": columns,
    }
