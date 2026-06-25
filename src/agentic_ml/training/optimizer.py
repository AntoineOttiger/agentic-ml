"""Optimisation d'hyperparamètres par Tree-structured Parzen Estimator (Optuna)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from sklearn.metrics import f1_score

from agentic_ml.config import (
    DEFAULT_F1_AVERAGE,
    DEFAULT_N_TRIALS,
    DEFAULT_RANDOM_SEED,
)
from agentic_ml.training.dataset import PreparedData, load_prepared_run
from agentic_ml.training.models import available_models, build_estimator
from agentic_ml.training.search_space import suggest_params, validate_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation : scores du meilleur modèle et sa configuration."""

    model_type: str
    best_params: dict[str, Any]
    train_f1: float
    eval_f1: float
    n_trials: int


class HyperparameterOptimizer:
    """Recherche d'hyperparamètres maximisant le F1 d'évaluation (set val)."""

    def __init__(
        self,
        *,
        f1_average: str = DEFAULT_F1_AVERAGE,
        random_seed: int = DEFAULT_RANDOM_SEED,
        n_trials: int = DEFAULT_N_TRIALS,
    ) -> None:
        self.f1_average = f1_average
        self.random_seed = random_seed
        self.n_trials = n_trials

    def _f1(self, estimator, X, y) -> float:
        return float(f1_score(y, estimator.predict(X), average=self.f1_average))

    def optimize(
        self,
        prepared_run: str | Path,
        model_type: str,
        search_space: dict[str, dict],
        *,
        n_trials: int | None = None,
    ) -> OptimizationResult:
        """Lance l'optimisation TPE et retourne le meilleur résultat.

        Args:
            prepared_run: identifiant ('001') ou chemin d'un run de `data/02_prepared`.
            model_type: nom du modèle (cf. `available_models()`).
            search_space: plages d'hyperparamètres (cf. `search_space` module).
            n_trials: nombre d'essais (défaut : valeur de l'instance).
        """
        if model_type not in available_models():
            raise ValueError(
                f"Modèle inconnu '{model_type}'. Disponibles : {available_models()}"
            )
        validate_search_space(search_space)

        data = load_prepared_run(prepared_run)
        n_trials = n_trials if n_trials is not None else self.n_trials

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(
            lambda trial: self._objective(trial, model_type, search_space, data),
            n_trials=n_trials,
        )

        # Réentraînement avec les meilleurs params pour obtenir train_f1 et eval_f1.
        best_estimator = build_estimator(
            model_type, study.best_params, random_state=self.random_seed
        )
        best_estimator.fit(data.X_train, data.y_train)

        return OptimizationResult(
            model_type=model_type,
            best_params=study.best_params,
            train_f1=self._f1(best_estimator, data.X_train, data.y_train),
            eval_f1=self._f1(best_estimator, data.X_val, data.y_val),
            n_trials=n_trials,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        model_type: str,
        search_space: dict[str, dict],
        data: PreparedData,
    ) -> float:
        params = suggest_params(trial, search_space)
        estimator = build_estimator(model_type, params, random_state=self.random_seed)
        estimator.fit(data.X_train, data.y_train)
        return self._f1(estimator, data.X_val, data.y_val)
