"""System prompt (mission) et construction du contexte injecté à chaque tour."""
from __future__ import annotations

import json

from agentic_ml.training.models import available_models

from agentic_ml.agents.training_agent.state import AgentState
from agentic_ml.agents.training_agent.tools import get_model_schema

SYSTEM_PROMPT = f"""\
Tu es un ingénieur en machine learning. Ton objectif est de MAXIMISER le F1-score \
d'évaluation (eval_f1) d'un modèle de classification, en choisissant un type de \
modèle parmi {", ".join(available_models())} et ses hyperparamètres.

Tu travailles par essais successifs. Le profil du dataset t'est fourni dès le départ. \
Avant chaque essai, tu formules UNE hypothèse explicite justifiant la configuration \
choisie, en t'appuyant sur l'historique des essais (trial_log) et sur le budget restant. \
Tu analyses l'écart train_f1 − eval_f1 comme signal d'overfitting : un écart important \
indique de réduire la complexité (régularisation, profondeur) plutôt que de l'augmenter.

Tu ne fais aucune transformation sur les données. Tu ne proposes que des hyperparamètres \
présents dans le schéma du modèle choisi, en respectant leurs bornes."""


STOP_SYSTEM_PROMPT = """\
Tu pilotes l'arrêt d'une recherche d'hyperparamètres dont l'objectif est de MAXIMISER \
eval_f1. À partir de l'historique des essais et du budget restant, tu décides s'il faut \
CONTINUER (un essai supplémentaire a une chance réelle d'améliorer le meilleur eval_f1) \
ou ARRÊTER (les gains stagnent, ou le résultat est déjà excellent).

Réponds par une décision « continue » ou « stop » et une justification courte. Le budget \
maximum est un plafond dur déjà appliqué séparément : tu n'as pas à t'en soucier, \
concentre-toi sur la pertinence de poursuivre la recherche."""


def _model_schemas() -> str:
    """Schémas des hyperparamètres valides de tous les modèles disponibles."""
    schemas = {
        m: get_model_schema(m)["hyperparameters"]
        for m in available_models()
    }
    return json.dumps(schemas, indent=2, ensure_ascii=False)


def build_context(state: AgentState) -> str:
    """Assemble le contexte dynamique pour le nœud propose_experiment."""
    runs_used = state.get("runs_used", 0)
    max_runs = state["max_runs"]
    budget_left = max_runs - runs_used

    best = state.get("best_trial")
    best_line = (
        f"{best['model_type']} eval_f1={best['eval_f1']:.4f}" if best else "aucun"
    )

    parts = [
        "## Profil du dataset",
        json.dumps(state["dataset_profile"], indent=2, ensure_ascii=False),
        "",
        "## Schémas des modèles (hyperparamètres valides et bornes)",
        _model_schemas(),
        "",
        "## Budget",
        f"runs utilisés : {runs_used} / {max_runs} — runs restants : {budget_left}",
        f"objectif : {state['objective']} (seuil cible eval_f1 >= {state['target_f1']})",
        f"meilleur essai à ce jour : {best_line}",
        "",
        "## Historique des essais (trial_log)",
        json.dumps(state.get("trial_log", []), indent=2, ensure_ascii=False),
    ]

    last_error = state.get("last_error")
    if last_error:
        parts += [
            "",
            "## ⚠️ Erreur de la proposition précédente (à corriger)",
            last_error,
        ]

    parts += [
        "",
        "Propose le prochain essai : une hypothèse explicite, un model_type, et un "
        "search_space (plages d'hyperparamètres) cohérent avec le schéma du modèle.",
    ]
    return "\n".join(parts)


def build_stop_context(state: AgentState) -> str:
    """Assemble le contexte pour la décision d'arrêt en mode « agent »."""
    runs_used = state.get("runs_used", 0)
    max_runs = state["max_runs"]
    budget_left = max_runs - runs_used

    best = state.get("best_trial")
    best_line = (
        f"{best['model_type']} eval_f1={best['eval_f1']:.4f}" if best else "aucun"
    )

    parts = [
        "## Budget",
        f"runs utilisés : {runs_used} / {max_runs} — runs restants : {budget_left}",
        f"objectif : {state['objective']}",
        f"meilleur essai à ce jour : {best_line}",
        "",
        "## Historique des essais (trial_log)",
        json.dumps(state.get("trial_log", []), indent=2, ensure_ascii=False),
        "",
        "Décide : faut-il CONTINUER (continue) ou ARRÊTER (stop) la recherche ?",
    ]
    return "\n".join(parts)
