"""Profilage d'un DataFrame en statistiques agrégées (sans données brutes).

L'Agent Analyse ne doit JAMAIS recevoir le DataFrame brut ni des séries de
valeurs numériques (coût en tokens + pollution du contexte). Ce module calcule
tout via pandas/numpy et ne renvoie qu'un dictionnaire de statistiques :
types, % de nulls, skewness, cardinalité, outliers (IQR), valeurs uniques des
catégorielles de faible cardinalité, corrélations entre features et avec la
target (si numérique).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from agentic_ml.config import MAX_CATEGORICAL_UNIQUES

# Seuils utilisés pour signaler des problèmes dans le profil.
HIGH_NULL_PCT = 40.0      # au-delà : la colonne est candidate à la suppression
SKEW_THRESHOLD = 1.0      # |skewness| au-delà : distribution asymétrique
HIGH_CARDINALITY = 10     # catégorielle au-delà : encoding non trivial
HIGH_CORRELATION = 0.95   # paires de features quasi colinéaires


def _column_profile(series: pd.Series) -> dict[str, Any]:
    """Profil d'une colonne : type, nulls, cardinalité, distribution, issues."""
    n = len(series)
    null_pct = round(float(series.isna().mean() * 100), 2) if n else 0.0
    n_unique = int(series.nunique(dropna=True))
    issues: list[str] = []
    if null_pct > HIGH_NULL_PCT:
        issues.append("high_nulls")

    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_pct": null_pct,
        "n_unique": n_unique,
        "skewness": None,
        "has_outliers": False,
        "issues": issues,
    }

    if ptypes.is_numeric_dtype(series):
        clean = series.dropna()
        if not clean.empty:
            profile.update(
                {
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "mean": round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                }
            )
            skew = float(clean.skew()) if len(clean) > 2 else 0.0
            profile["skewness"] = round(skew, 4)
            if abs(skew) > SKEW_THRESHOLD:
                issues.append("skewed")
            has_outliers, n_outliers = _iqr_outliers(clean)
            profile["has_outliers"] = has_outliers
            profile["n_outliers"] = n_outliers
            if has_outliers:
                issues.append("outliers")
    else:
        # Catégorielle / texte : exposer les valeurs uniques seulement si faible cardinalité.
        if n_unique <= MAX_CATEGORICAL_UNIQUES:
            profile["unique_values"] = [
                str(v) for v in series.dropna().unique().tolist()
            ]
        if n_unique >= HIGH_CARDINALITY:
            issues.append("high_cardinality")

    return profile


def _iqr_outliers(series: pd.Series) -> tuple[bool, int]:
    """Détection d'outliers par la méthode IQR. Renvoie ``(présence, nombre)``."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return False, 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((series < low) | (series > high)).sum())
    return n_outliers > 0, n_outliers


def _correlations(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Corrélations numériques : paires fortement corrélées + corrélation à la target."""
    numeric = df.select_dtypes(include=np.number)
    out: dict[str, Any] = {"highly_correlated_pairs": [], "target_correlation": {}}
    if numeric.shape[1] < 2:
        return out

    corr = numeric.corr(numeric_only=True)

    # Paires de features (hors target) quasi colinéaires.
    features = [c for c in corr.columns if c != target_column]
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) > HIGH_CORRELATION:
                out["highly_correlated_pairs"].append(
                    {"columns": [a, b], "correlation": round(float(value), 4)}
                )

    # Corrélation à la target uniquement si la target est numérique.
    if target_column in corr.columns:
        target_corr = corr[target_column].drop(labels=[target_column], errors="ignore")
        out["target_correlation"] = {
            col: round(float(val), 4)
            for col, val in target_corr.items()
            if pd.notna(val)
        }

    return out


def profile_dataframe(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Construit le profil statistique complet du DataFrame.

    Args:
        df: DataFrame courant.
        target_column: nom de la colonne cible.

    Returns:
        Dictionnaire JSON-sérialisable : aucune ligne ni série brute, uniquement
        des statistiques agrégées.
    """
    columns = {col: _column_profile(df[col]) for col in df.columns}

    global_issues: list[str] = []
    if df.isna().any().any():
        global_issues.append("missing_values_present")
    if df.duplicated().any():
        global_issues.append(f"duplicate_rows({int(df.duplicated().sum())})")

    correlations = _correlations(df, target_column)
    if correlations["highly_correlated_pairs"]:
        global_issues.append("collinear_features")

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "target_is_numeric": bool(
            target_column in df.columns and ptypes.is_numeric_dtype(df[target_column])
        ),
        "columns": columns,
        "correlations": correlations,
        "global_issues": global_issues,
    }
