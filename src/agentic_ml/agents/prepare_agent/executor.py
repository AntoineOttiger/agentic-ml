"""Exécution déterministe des scripts générés par l'agent (§5).

> L'agent décide ; le code exécute. Aucun agent ne touche directement au dataframe.

Le script de l'agent doit définir `transform(df: pd.DataFrame) -> pd.DataFrame`.
Il est compilé puis exécuté **en mémoire**, dans un espace de noms fournissant
`pd`/`np`/`re`/`math`/`datetime`. La garantie anti-leakage ne repose jamais sur
cette exécution mais sur le gate d'invariance (§6), appliqué en amont.
"""
from __future__ import annotations

import datetime as _datetime
import math
import re
import traceback
from typing import Callable

import numpy as np
import pandas as pd

TransformFn = Callable[[pd.DataFrame], pd.DataFrame]

#: Symboles mis à disposition du script de l'agent.
_TRANSFORM_GLOBALS = {
    "pd": pd,
    "np": np,
    "re": re,
    "math": math,
    "datetime": _datetime,
}


class ExecutorError(RuntimeError):
    """Le script est invalide, ne définit pas `transform`, ou lève à l'exécution."""


def compile_transform(code: str) -> TransformFn:
    """Compile le script et renvoie la fonction `transform`.

    Lève `ExecutorError` (avec contexte) si le code ne compile pas ou ne définit
    pas un callable `transform`.
    """
    namespace: dict = dict(_TRANSFORM_GLOBALS)
    try:
        compiled = compile(code, "<agent_transform>", "exec")
        exec(compiled, namespace)  # noqa: S102 — exécution voulue, cf. §5
    except SyntaxError as exc:
        raise ExecutorError(f"Erreur de syntaxe dans le script : {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ExecutorError(f"Le script a échoué au chargement : {exc}") from exc

    fn = namespace.get("transform")
    if not callable(fn):
        raise ExecutorError(
            "Le script doit définir une fonction `transform(df: pd.DataFrame) -> pd.DataFrame`."
        )
    return fn


def apply_transform(fn: TransformFn, df: pd.DataFrame) -> pd.DataFrame:
    """Applique `fn` à une **copie** de `df` et valide grossièrement la sortie.

    Lève `ExecutorError` avec la traceback complète en cas d'erreur d'exécution —
    ce message est renvoyé à l'agent pour qu'il corrige (§6.4).
    """
    try:
        out = fn(df.copy())
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        raise ExecutorError(f"`transform` a levé une exception : {exc}\n{tb}") from exc

    if not isinstance(out, pd.DataFrame):
        raise ExecutorError(
            f"`transform` doit renvoyer un DataFrame, pas {type(out).__name__}."
        )
    return out
