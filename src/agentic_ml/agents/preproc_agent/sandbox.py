"""Exécution sandboxée du code Python généré par les agents.

Le code généré manipule un DataFrame nommé `df` (et éventuellement une variable
`target`). Il est exécuté dans un namespace isolé et toujours sur une *copie* :
en cas d'erreur, le DataFrame original est conservé intact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def execute_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Exécute `code` dans un namespace isolé et renvoie ``(df_modifié, erreur)``.

    On utilise un namespace unique (``globals`` == ``locals``) afin d'éviter le
    bug de portée des list-comprehensions / fonctions imbriquées qui survient
    quand ``exec`` reçoit des dictionnaires globals et locals distincts. Le code
    importe lui-même sklearn / scipy au besoin (``__builtins__`` est injecté
    automatiquement par ``exec``).

    Args:
        code: code Python pandas/sklearn valide manipulant `df`.
        df: DataFrame d'entrée (copié avant exécution).

    Returns:
        ``(df_modifié, None)`` en cas de succès, ``(df_original, message)`` sinon.
    """
    namespace: dict = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, namespace)
        result = namespace.get("df")
        if not isinstance(result, pd.DataFrame):
            return df, "Le code n'a pas laissé un pandas.DataFrame dans la variable `df`."
        return result, None
    except Exception as exc:  # noqa: BLE001 — on remonte l'erreur au LLM pour correction
        return df, f"{type(exc).__name__}: {exc}"
