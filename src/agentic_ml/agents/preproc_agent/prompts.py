"""Prompts système et constructeurs de contexte des agents preproc.

Les prompts système reprennent les préceptes de `notes/AGENTS_PREPROC_FE.md`.
Les contextes utilisateur sont assemblés dynamiquement à chaque tour, à partir
du state, et n'exposent jamais le DataFrame brut (uniquement des statistiques).
"""
from __future__ import annotations

import json
from typing import Any

from agentic_ml.agents.preproc_agent.profiling import profile_dataframe
from agentic_ml.agents.preproc_agent.state import MLPipelineState

ANALYSIS_SYSTEM_PROMPT = """\
Tu es un expert en data science. Ton rôle est d'analyser le profil statistique \
d'un DataFrame pandas et de produire un rapport structuré.

Pour chaque colonne, en t'appuyant UNIQUEMENT sur les statistiques fournies, évalue :
- type de données (numérique, catégorielle, datetime, texte)
- pourcentage de valeurs manquantes
- distribution (skewness, outliers, min/max)
- cardinalité pour les catégorielles
- corrélation avec la target (si fournie)

Recommande des actions concrètes et NON REDONDANTES. Ne recommande jamais une \
transformation déjà présente dans `applied_transformations` ni une feature déjà \
dans `created_features`. Si les données sont propres et qu'aucune action utile ne \
reste à faire, mets `is_clean` à true et renvoie une liste d'actions vide.

Chaque action a un `type` ("preprocessing" ou "feature_engineering"), un `action` \
(ex: "impute_median", "log_transform", "create_ratio"), les `columns` concernées \
et une `reason` courte."""

PREPROCESSING_SYSTEM_PROMPT = """\
Tu es un expert ML spécialisé en preprocessing de données.
Tu reçois un rapport d'analyse et UNE action à appliquer.

Respecte ces préceptes de bonnes pratiques ML :
- Valeurs manquantes : médiane pour les numériques, mode pour les catégorielles.
  Si null > 40% : supprimer la colonne.
- Outliers : méthode IQR, cap (clip) plutôt que suppression sauf cas extrêmes.
- Distributions skewed (|skewness| > 1) : log1p si valeurs positives, Box-Cox sinon.
- Catégorielles faible cardinalité (< 10) : One-Hot Encoding.
- Catégorielles haute cardinalité (>= 10) : Target Encoding ou Ordinal selon le contexte.
- Features corrélées entre elles (> 0.95) : supprimer la moins corrélée à la target.
- Standardisation (StandardScaler) en dernier, après toutes les autres transformations.

Ne refais jamais une transformation déjà listée dans `applied_transformations`.
Génère uniquement du code Python pandas/sklearn valide. Importe toi-même les modules \
nécessaires (pandas est `pd`, numpy est `np`, déjà disponibles). Le DataFrame s'appelle \
`df` et la colonne cible est désignée par la variable `target` (son nom). Ne touche pas \
à la colonne cible. Le code doit réaffecter le résultat à `df`. Aucune E/S disque."""

FEATURE_ENGINEERING_SYSTEM_PROMPT = """\
Tu es un expert ML spécialisé en feature engineering.
Tu reçois un rapport d'analyse et UNE action à effectuer.

Respecte ces préceptes :
- Crée des ratios entre features liées (ex: prix/surface → prix_m2).
- Extrait des composantes temporelles depuis les datetimes (jour, mois, heure, etc.).
- Crée des features d'interaction entre variables fortement corrélées à la target.
- Applique des transformations polynomiales (degré 2 max) sur les features importantes.
- Toute nouvelle feature numérique doit être normalisée.
- Documente chaque feature créée avec un nom explicite (snake_case).

Ne recrée jamais une feature déjà listée dans `created_features`.
Génère uniquement du code Python pandas valide. Importe toi-même les modules \
nécessaires (pandas est `pd`, numpy est `np`, déjà disponibles). Le DataFrame s'appelle \
`df` et la colonne cible est désignée par la variable `target` (son nom). Ne touche pas \
à la colonne cible. Le code doit réaffecter le résultat à `df`. Aucune E/S disque."""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def build_analysis_context(state: MLPipelineState) -> str:
    """Contexte utilisateur de l'Agent Analyse : profil statistique + historique."""
    df = state["dataframe"]
    target = state["target_column"]
    profile = profile_dataframe(df, target)

    parts = [
        "## Profil statistique du DataFrame (agrégé, aucune donnée brute)",
        _dumps(profile),
        "",
        f"## Colonne cible\n{target}",
        "",
        "## Transformations déjà appliquées (ne pas répéter)",
        _dumps(state.get("applied_transformations", [])),
        "",
        "## Features déjà créées (ne pas répéter)",
        _dumps(state.get("created_features", [])),
        "",
        "Produis le rapport d'analyse structuré.",
    ]
    return "\n".join(parts)


def _action_context(state: MLPipelineState, action: dict[str, Any], history_key: str) -> str:
    """Contexte commun aux agents preproc/FE : rapport + action courante + historique."""
    parts = [
        "## Rapport d'analyse",
        _dumps(state.get("analysis_report", {})),
        "",
        "## Action à appliquer (une seule)",
        _dumps(action),
        "",
        f"## Colonne cible (variable `target`)\n{state['target_column']}",
        "",
        "## Historique (ne pas répéter)",
        _dumps(state.get(history_key, [])),
        "",
        "Génère le code Python qui applique cette action, et la liste `applied` "
        "des libellés correspondants.",
    ]
    return "\n".join(parts)


def build_preproc_context(state: MLPipelineState, action: dict[str, Any]) -> str:
    """Contexte de l'Agent Preprocessing pour une action donnée."""
    return _action_context(state, action, "applied_transformations")


def build_fe_context(state: MLPipelineState, action: dict[str, Any]) -> str:
    """Contexte de l'Agent Feature Engineering pour une action donnée."""
    return _action_context(state, action, "created_features")


def build_retry_suffix(previous_code: str, error: str) -> str:
    """Bloc ajouté au contexte pour demander la correction d'un code en échec."""
    return "\n".join(
        [
            "",
            "## Le code précédent a échoué — corrige-le",
            "### Code précédent",
            "```python",
            previous_code,
            "```",
            "### Erreur",
            error,
            "",
            "Renvoie une version corrigée du code.",
        ]
    )
