"""Traduction d'un espace de recherche déclaratif en suggestions Optuna.

Format de spec par hyperparamètre :

    {
      "n_estimators":  {"type": "int",   "low": 50, "high": 500, "step": 10},
      "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": true},
      "kernel":        {"type": "categorical", "choices": ["rbf", "linear"]}
    }
"""
from __future__ import annotations

from typing import Any

import optuna

_NUMERIC_TYPES = {"int", "float"}


def validate_search_space(search_space: dict[str, dict]) -> None:
    """Vérifie la structure de l'espace de recherche, erreur lisible sinon."""
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space doit être un dict non vide.")

    for name, spec in search_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"'{name}': chaque spec doit être un dict avec une clé 'type'.")

        ptype = spec["type"]
        if ptype in _NUMERIC_TYPES:
            missing = [k for k in ("low", "high") if k not in spec]
            if missing:
                raise ValueError(
                    f"'{name}' (type {ptype}): clés manquantes {missing} (attendu 'low' et 'high')."
                )
        elif ptype == "categorical":
            if not spec.get("choices"):
                raise ValueError(f"'{name}' (categorical): clé 'choices' requise et non vide.")
        else:
            raise ValueError(
                f"'{name}': type '{ptype}' inconnu (attendu 'int', 'float' ou 'categorical')."
            )


def suggest_params(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """Échantillonne un jeu d'hyperparamètres pour un trial donné."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"],
                step=spec.get("step", 1), log=spec.get("log", False),
            )
        elif ptype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"],
                step=spec.get("step"), log=spec.get("log", False),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:  # pragma: no cover - couvert par validate_search_space
            raise ValueError(f"'{name}': type '{ptype}' inconnu.")
    return params
