# Implémentation : Agents Preprocessing & Feature Engineering

## Contexte

Ce document décrit l'implémentation d'un système agentique **indépendant et isolé**, dédié au preprocessing et au feature engineering. Il s'exécute séparément du système d'entraînement (déjà implémenté) — les deux systèmes ne partagent ni state, ni graphe, ni agents.

Ce système comprend trois agents :

- **Agent Analyse** : profile l'état courant des données
- **Agent Preprocessing** : applique les transformations de nettoyage
- **Agent Feature Engineering** : crée et transforme les features

Le projet utilise LangChain/LangGraph — respecter les conventions et patterns déjà en place dans le codebase.

Les données d'entrée viennent d'un .csv dans le dossier data/00_raw

Les données finales transformées par le système agentique doivent être enregistré dans data/01_preproc


---

## Architecture cible

```
Orchestrateur (LangGraph StateGraph)
       │
       ▼
Agent Analyse  ──→  Agent Preprocessing  ──→  Agent Feature Engineering
       ▲                    │                          │
       └────────────────────┴──────────────────────────┘
                     (boucle itérative)
```

L'orchestrateur tourne en boucle jusqu'à ce que l'Agent Analyse ne détecte plus d'action à effectuer.

---

## State partagé

Définir un `TypedDict` pour le state LangGraph partagé entre les agents :

```python
class MLPipelineState(TypedDict):
    # Données
    dataframe: Any                        # pd.DataFrame courant
    target_column: str                    # Nom de la colonne cible

    # Historique des transformations (éviter les doublons)
    applied_transformations: list[str]    # Ex: ["log(price)", "drop(id)", ...]
    created_features: list[str]           # Ex: ["prix_au_m2", "age_annees"]

    # Communication entre agents
    analysis_report: str                  # Rapport JSON de l'Agent Analyse
    actions_to_apply: list[dict]          # Actions décidées par l'orchestrateur
    generated_code: str                   # Code Python généré par les agents
    execution_error: str | None           # Erreur d'exécution si applicable

    # Contrôle de boucle
    iteration_count: int
    max_iterations: int                   # Limite de sécurité (défaut: 10)
    should_continue: bool
```

---

## Agent Analyse

### Rôle

Profiler les données et produire un rapport structuré JSON décrivant l'état courant. C'est lui qui "voit" les données et informe les autres agents.

### Prompt système

```
Tu es un expert en data science. Ton rôle est d'analyser un DataFrame pandas
et de produire un rapport JSON structuré.

Pour chaque colonne, analyse :
- type de données (numérique, catégorielle, datetime, texte)
- pourcentage de valeurs manquantes
- distribution (skewness, outliers, min/max)
- cardinalité pour les catégorielles
- corrélation avec la target

Retourne UNIQUEMENT un JSON valide avec cette structure :
{
  "columns": {
    "<col_name>": {
      "dtype": str,
      "null_pct": float,
      "skewness": float | null,
      "n_unique": int,
      "has_outliers": bool,
      "issues": [str]   // ex: ["high_nulls", "skewed", "high_cardinality"]
    }
  },
  "global_issues": [str],
  "recommended_actions": [
    {
      "type": "preprocessing" | "feature_engineering",
      "action": str,         // ex: "impute_median", "log_transform", "create_ratio"
      "columns": [str],
      "reason": str
    }
  ],
  "is_clean": bool           // true si aucune action n'est nécessaire
}
```

### Implémentation

- Calcule les stats avec pandas/numpy **avant** d'appeler le LLM
- Passe ces stats dans le prompt utilisateur (pas en tool call)
- Le LLM interprète les stats et génère les `recommended_actions`
- Si `is_clean: true` → mettre `should_continue = False` dans le state

### Contrainte : pas de données brutes dans le contexte

La raison est uniquement le **coût en tokens et la pollution du contexte** — pas la confidentialité.

Ce qui est **interdit** dans le prompt :

- Valeurs brutes d'une colonne numérique (`[1.2, 3.4, 5.6, ...]`)
- Lignes ou échantillons du DataFrame (`.head()`, `.sample()`)

Ce qui est **autorisé** :

- Statistiques agrégées : mean, median, std, min, max, skewness, kurtosis
- Pourcentage de nulls, cardinalité
- Corrélations numériques avec la target
- Types de données, noms de colonnes
- **Valeurs uniques des colonnes catégorielles** — utiles pour décider du bon encoding (ex: détecter une colonne ordinale, des booléens déguisés, des fautes de frappe). Les passer uniquement si cardinalité <= 50, sinon passer uniquement le nombre.

L'Agent Analyse calcule tout via pandas/numpy et ne transmet au LLM qu'un dictionnaire de statistiques — jamais le DataFrame brut ni des séries de valeurs numériques.

---

## Agent Preprocessing

### Rôle

Prendre les actions de type `preprocessing` du rapport d'analyse et générer + exécuter le code Python correspondant.

### Prompt système

```
Tu es un expert ML spécialisé en preprocessing de données.
Tu reçois un rapport d'analyse et une liste d'actions à appliquer.

Respecte ces préceptes de bonnes pratiques ML :
- Valeurs manquantes : médiane pour les numériques, mode pour les catégorielles.
  Si null > 40% : supprimer la colonne.
- Outliers : IQR method, cap plutôt que suppression sauf cas extrêmes.
- Distributions skewed (|skewness| > 1) : log1p si valeurs positives, Box-Cox sinon.
- Catégorielles faible cardinalité (< 10) : One-Hot Encoding.
- Catégorielles haute cardinalité (>= 10) : Target Encoding ou Ordinal selon le contexte.
- Features corrélées entre elles (> 0.95) : supprimer la moins corrélée à la target.
- Standardisation (StandardScaler) en dernier, après toutes les autres transformations.

Ne refais jamais une transformation déjà listée dans applied_transformations.
Génère uniquement du code Python pandas/sklearn valide.
Le DataFrame s'appelle `df` et la target s'appelle `target`.
```

### Implémentation

- Le code généré est exécuté dans un environnement sandboxé (voir section Exécution)
- Mettre à jour `applied_transformations` dans le state après chaque exécution réussie
- En cas d'erreur : repasser le code + l'erreur au LLM pour correction (max 3 retry)

---

## Agent Feature Engineering

### Rôle

Prendre les actions de type `feature_engineering` et créer de nouvelles features pertinentes.

### Prompt système

```
Tu es un expert ML spécialisé en feature engineering.
Tu reçois un rapport d'analyse et une liste d'actions à effectuer.

Respecte ces préceptes :
- Crée des ratios entre features liées (ex: prix/surface → prix_m2).
- Extrait des composantes temporelles depuis les datetimes (jour, mois, heure, etc.).
- Crée des features d'interaction entre variables fortement corrélées à la target.
- Applique des transformations polynomiales (degré 2 max) sur les features importantes.
- Toute nouvelle feature numérique doit être normalisée.
- Documente chaque feature créée avec un nom explicite (snake_case).

Ne recrée jamais une feature déjà listée dans created_features.
Génère uniquement du code Python pandas valide.
Le DataFrame s'appelle `df`.
```

### Implémentation

- Même pattern d'exécution que l'Agent Preprocessing
- Mettre à jour `created_features` dans le state après chaque exécution réussie

---

## Exécution de code (Sandboxing)

Créer un utilitaire `execute_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]` :

```python
def execute_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Exécute le code généré dans un namespace isolé.
    Retourne (df_modifié, erreur_ou_None).
    """
    local_ns = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, {}, local_ns)
        return local_ns["df"], None
    except Exception as e:
        return df, str(e)
```

**Important** : toujours travailler sur une copie du DataFrame. En cas d'erreur, le DataFrame original est conservé.

---

## Orchestrateur (LangGraph)

### Structure du graphe

```python
graph = StateGraph(MLPipelineState)

graph.add_node("analyse", agent_analyse)
graph.add_node("preprocessing", agent_preprocessing)
graph.add_node("feature_engineering", agent_feature_engineering)

graph.set_entry_point("analyse")

graph.add_conditional_edges(
    "analyse",
    route_after_analysis,       # Fonction de routing (voir ci-dessous)
    {
        "preprocessing": "preprocessing",
        "feature_engineering": "feature_engineering",
        "end": END,
    }
)

graph.add_edge("preprocessing", "analyse")       # Retour à l'analyse après chaque action
graph.add_edge("feature_engineering", "analyse")
```

### Fonction de routing

```python
def route_after_analysis(state: MLPipelineState) -> str:
    if not state["should_continue"]:
        return "end"
    if state["iteration_count"] >= state["max_iterations"]:
        return "end"

    actions = state["actions_to_apply"]
    if any(a["type"] == "preprocessing" for a in actions):
        return "preprocessing"
    if any(a["type"] == "feature_engineering" for a in actions):
        return "feature_engineering"
    return "end"
```

**Note** : traiter une action à la fois par itération pour maintenir une traçabilité fine.

---

## Critères d'arrêt

Le système s'arrête quand l'une de ces conditions est vraie :

- `is_clean: true` dans le rapport de l'Agent Analyse
- `iteration_count >= max_iterations`
- Toutes les `recommended_actions` ont été appliquées (vérifier via `applied_transformations` et `created_features`)

---

## Points d'attention

- **Ne pas réinitialiser le state** entre les itérations — l'historique des transformations est critique
- **Logger chaque transformation** avec suffisamment de détail pour être reproductible
- **Exposer le DataFrame final** et la liste complète des transformations appliquées en sortie du graphe
- **Tester avec des DataFrames variés** : nulls, catégorielles, datetimes, distributions skewed
