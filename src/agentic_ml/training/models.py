"""Registre des modèles disponibles pour l'optimisation.

Point d'extension principal : ajouter un modèle = ajouter une entrée à
``MODEL_REGISTRY``. L'optimiseur ne connaît jamais les modèles concrets, seulement
ce registre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    """Description d'un modèle pris en charge.

    Attributes:
        builder: fabrique un estimateur sklearn-compatible à partir des params
            variables (issus d'un trial) et des params fixes (seed...).
        needs_label_encoding: True si la cible doit être encodée en entiers
            (requis par XGBoost, neutre pour RF/SVM).
        default_search_space: espace de recherche de référence (sert de schéma,
            base future du tool `get_model_schema`).
    """

    builder: Callable[..., Any]
    needs_label_encoding: bool = False
    default_search_space: dict[str, dict] = field(default_factory=dict)


def _build_xgboost(params: dict, *, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def _build_random_forest(params: dict, *, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params,
        random_state=random_state,
        n_jobs=-1,
    )


def _build_svm(params: dict, *, random_state: int) -> SVC:
    # SVC n'expose pas n_jobs ; random_state n'intervient que pour la shrinking
    # heuristic / probability=True, mais on le passe pour la reproductibilité.
    return SVC(**params, random_state=random_state)


def _build_logistic_regression(params: dict, *, random_state: int) -> Pipeline:
    # Modèle linéaire sensible à l'échelle → StandardScaler en amont.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**params, max_iter=1000, random_state=random_state),
    )


def _build_knn(params: dict, *, random_state: int) -> Pipeline:
    # KNN n'utilise pas random_state ; signature homogène. Distances → scaling requis.
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))


def _build_gaussian_nb(params: dict, *, random_state: int) -> GaussianNB:
    # GaussianNB n'utilise pas random_state ; signature homogène.
    return GaussianNB(**params)


def _build_mlp(params: dict, *, random_state: int) -> Pipeline:
    # hidden_layer_sizes arrive en chaîne ("50" ou "50,50") → tuple d'entiers.
    params = dict(params)
    hls = params.get("hidden_layer_sizes")
    if isinstance(hls, str):
        params["hidden_layer_sizes"] = tuple(int(n) for n in hls.split(","))
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(**params, max_iter=1000, random_state=random_state),
    )


def _build_lda(params: dict, *, random_state: int) -> LinearDiscriminantAnalysis:
    # LDA n'utilise pas random_state ; signature homogène.
    return LinearDiscriminantAnalysis(**params)


def _build_extra_trees(params: dict, *, random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**params, random_state=random_state, n_jobs=-1)


def _build_hist_gradient_boosting(
    params: dict, *, random_state: int
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**params, random_state=random_state)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        builder=_build_xgboost,
        needs_label_encoding=True,
        default_search_space={
            "n_estimators":  {"type": "int",   "low": 50, "high": 500, "step": 10},
            "max_depth":     {"type": "int",   "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample":     {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        },
    ),
    "random_forest": ModelSpec(
        builder=_build_random_forest,
        needs_label_encoding=False,
        default_search_space={
            "n_estimators":      {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth":         {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
            "min_samples_leaf":  {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
        },
    ),
    "svm": ModelSpec(
        builder=_build_svm,
        needs_label_encoding=False,
        default_search_space={
            "C":      {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "gamma":  {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
        },
    ),
    "logistic_regression": ModelSpec(
        builder=_build_logistic_regression,
        needs_label_encoding=False,
        default_search_space={
            "C":      {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
            "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
        },
    ),
    "k_nearest_neighbors": ModelSpec(
        builder=_build_knn,
        needs_label_encoding=False,
        default_search_space={
            "n_neighbors": {"type": "int", "low": 1, "high": 30},
            "weights":     {"type": "categorical", "choices": ["uniform", "distance"]},
            "p":           {"type": "int", "low": 1, "high": 2},
        },
    ),
    "gaussian_nb": ModelSpec(
        builder=_build_gaussian_nb,
        needs_label_encoding=False,
        default_search_space={
            "var_smoothing": {"type": "float", "low": 1e-12, "high": 1e-3, "log": True},
        },
    ),
    "mlp": ModelSpec(
        builder=_build_mlp,
        needs_label_encoding=False,
        default_search_space={
            # Chaînes converties en tuples par le builder (cf. _build_mlp).
            "hidden_layer_sizes": {
                "type": "categorical",
                "choices": ["50", "100", "50,50", "100,50"],
            },
            "alpha":              {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        },
    ),
    "lda": ModelSpec(
        builder=_build_lda,
        needs_label_encoding=False,
        default_search_space={
            # shrinkage écarté : incompatible avec solver='svd' → risque de crash.
            "solver": {"type": "categorical", "choices": ["svd", "lsqr"]},
        },
    ),
    "extra_trees": ModelSpec(
        builder=_build_extra_trees,
        needs_label_encoding=False,
        default_search_space={
            "n_estimators":      {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth":         {"type": "int", "low": 2, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
            "min_samples_leaf":  {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
        },
    ),
    "hist_gradient_boosting": ModelSpec(
        builder=_build_hist_gradient_boosting,
        needs_label_encoding=False,
        default_search_space={
            "max_iter":          {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth":         {"type": "int", "low": 2, "high": 12},
            "learning_rate":     {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "l2_regularization": {"type": "float", "low": 0.0, "high": 1.0},
        },
    ),
}


def available_models() -> list[str]:
    """Noms des modèles enregistrés."""
    return list(MODEL_REGISTRY)


def get_model_spec(model_type: str) -> ModelSpec:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Modèle inconnu '{model_type}'. Disponibles : {available_models()}"
        )
    return MODEL_REGISTRY[model_type]


def build_estimator(model_type: str, params: dict, *, random_state: int) -> Any:
    """Instancie un estimateur du modèle demandé avec ses hyperparamètres."""
    spec = get_model_spec(model_type)
    return spec.builder(params, random_state=random_state)
