# Architecture de l'agent — Recherche ML automatisée (proto)

## Scope du prototype

L'agent ne fait **aucune transformation sur les données**. Il opère une boucle de recherche dirigée dont les seules décisions sont :

1. choisir un **modèle** (`XGBoost`, `Random Forest`, `SVM`),
2. choisir les **hyperparamètres** de ce modèle,
3. décider si l'optimisation est **suffisante** (continuer / arrêter).

## Objectif

**Maximiser le F1-score d'évaluation (`eval_f1`).**

La pipeline ML renvoie à chaque run le F1-score sur le train (`train_f1`) et sur l'éval (`eval_f1`). La métrique optimisée est `eval_f1` (sens : maximisation). L'écart `train_f1 − eval_f1` est un **signal d'overfitting** : un écart important indique de réduire la complexité du modèle (régularisation, profondeur, etc.) plutôt que de l'augmenter.

## Mission (system prompt)

> Tu es un ingénieur en machine learning. Ton objectif est de **maximiser le F1-score d'évaluation** d'un modèle de classification, en choisissant un type de modèle parmi `XGBoost`, `Random Forest`, `SVM` et ses hyperparamètres.
>
> Tu travailles par essais successifs. Le profil du dataset t'est fourni dès le départ. Avant chaque essai, tu formules **une hypothèse explicite** justifiant la configuration choisie, en t'appuyant sur l'historique des essais déjà réalisés (`trial_log`) et sur le budget restant. Après chaque run, tu analyses le résultat — en particulier l'écart `train_f1 − eval_f1` comme signal d'overfitting — puis tu décides de continuer ou d'arrêter selon les critères d'arrêt.
>
> Tu ne fais aucune transformation sur les données.

## Contexte injecté dès l'initialisation

Ces informations sont statiques et toujours nécessaires : elles sont placées dans le contexte initial plutôt que récupérées par tool call.

* **Profil du dataset** (`dataset_profile`) : taille, noms et ranges des features, noms et ranges des outputs, type de tâche (classification), distribution des classes.
* **Budget total** : nombre maximum de runs de pipeline autorisés (`max_runs`).
* **Critères d'arrêt** (voir plus bas).

## État de l'agent (state)

L'état porte la mémoire de la recherche, réinjectée à chaque tour :

```
state = {
  "dataset_profile": {...},          # injecté au départ, immuable
  "objective": "maximize eval_f1",
  "max_runs": int,
  "runs_used": int,                  # incrémenté à chaque pipeline
  "trial_log": [                     # historique structuré des essais
    {
      "run_id": int,
      "hypothesis": str,             # justification écrite avant le run
      "model_type": str,
      "hyperparameters": {...},
      "train_f1": float,
      "eval_f1": float,
    },
    ...
  ],
  "best_trial": {...},               # meilleur essai à ce jour
  "decision": "continue" | "stop",
}
```

## Boucle agentique (nœuds)

La décision « proposer un essai » est séparée de la décision « arrêter », car l'auto-terminaison est ce qu'un LLM gère le plus mal (arrêt prématuré ou boucle infinie).

1. **`propose_experiment`** — à partir du `trial_log` et du budget restant, formule une **hypothèse explicite** puis une configuration `(model_type, hyperparameters)`.
2. **`run_pipeline`** — appelle `launch_ml_pipeline`, valide les hyperparamètres, enregistre le résultat dans le `trial_log`, met à jour `runs_used` et `best_trial`.
3. **`evaluate_stop`** — ne regarde que le `trial_log` et les critères d'arrêt ; renvoie `continue` ou `stop`. Si `continue`, retour au nœud 1.

## Critères d'arrêt (chiffrés)

L'agent s'arrête dès qu'**une** des conditions est remplie :

* **Budget épuisé** : `runs_used >= max_runs`.
* **Convergence** : amélioration de `eval_f1` < `ε` sur les `K` derniers essais.
* **Seuil cible atteint** : `eval_f1 >= target_f1`.

> Le **budget restant** (`max_runs − runs_used`) est explicitement injecté dans le contexte à chaque tour : un agent qui sait qu'il lui reste 2 runs sur 10 décide différemment d'un agent aveugle à son budget.

## Tools

### `get_dataset_profile() -> profile`
Renvoie en **un seul appel** l'ensemble des métadonnées du dataset (taille, noms/ranges des features et outputs, type de tâche, distribution des classes). Remplace les 5 appels séquentiels précédents. *En pratique, ce profil est injecté dès l'initialisation ; le tool reste disponible si l'agent a besoin de le re-consulter.*

### `get_model_schema(model_type) -> schema`
Renvoie un **dict typé** décrivant les hyperparamètres valides du modèle demandé : noms, valeurs par défaut, bornes/ranges autorisés, type. Source fiable et testable sans LLM, pour les faits structurés (les espaces d'hyperparamètres de XGBoost, RF et SVM n'ont presque rien en commun). À utiliser pour cadrer toute proposition de configuration.

### `query_ml_knowledge_base(request) -> response`
Système RAG renvoyant le chunk le plus pertinent. Réservé au **savoir libre / qualitatif** : « quel modèle pour des classes déséquilibrées », « comment réduire l'overfitting d'un SVM », etc. **N'est pas utilisé** pour récupérer des noms/ranges d'hyperparamètres précis (→ `get_model_schema`).

### `launch_ml_pipeline(model_type, hyperparameters) -> {train_f1, eval_f1}`
Lance la pipeline d'entraînement et renvoie le F1-score sur le train et sur l'éval.
**Validation préalable** : les hyperparamètres sont vérifiés contre le schéma du modèle. Si un nom est inconnu ou une valeur hors-range, le tool renvoie une **erreur structurée** que l'agent peut lire et corriger, plutôt qu'un crash ou un comportement silencieux.

## Observabilité

Tracer pour chaque tour : le raisonnement / l'hypothèse, le tool call et ses arguments, le résultat, et la table d'essais courante (LangSmith ou logging maison). Sur un proto de boucle agentique, pouvoir **rejouer une trajectoire** est essentiel pour débugger les décisions d'arrêt.

## Hors scope (notes pour la v2)

* Transformations / feature engineering sur les données.
* `launch_ml_pipeline` acceptant un **batch de configs en parallèle** pour accélérer l'exploration.
* Métriques multiples / optimisation multi-objectif.
