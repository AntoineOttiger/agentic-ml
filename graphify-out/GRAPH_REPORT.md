# Graph Report - agentic-ml  (2026-06-26)

## Corpus Check
- 40 files · ~11,582 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 288 nodes · 459 edges · 21 communities (17 shown, 4 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 48 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4a9fa3be`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 22|Community 22]]

## God Nodes (most connected - your core abstractions)
1. `AgentState` - 14 edges
2. `MLPipelineState` - 14 edges
3. `run_agent()` - 10 edges
4. `launch_ml_pipeline()` - 9 edges
5. `HyperparameterOptimizer` - 9 edges
6. `_apply_action()` - 9 edges
7. `run_preproc_agent()` - 8 edges
8. `agent_preprocessing()` - 8 edges
9. `agent_feature_engineering()` - 8 edges
10. `available_models()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `parse_args()` --calls--> `available_models()`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/models.py
- `main()` --calls--> `HyperparameterOptimizer`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/optimizer.py
- `main()` --calls--> `run_agent()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/graph.py
- `main()` --calls--> `format_class_report()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/report.py
- `main()` --calls--> `run_preproc_agent()`  [INFERRED]
  scripts/run_preproc_agent.py → src/agentic_ml/agents/preproc_agent/graph.py

## Import Cycles
- None detected.

## Communities (21 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (33): build_agent_graph(), Assemblage du StateGraph et point d'entrée `run_agent`., Construit et compile le graphe de la boucle de recherche.      Flux : propose_ex, Exécute la boucle agentique de bout en bout et renvoie l'état final.      Args:, run_agent(), evaluate_stop(), _evaluate_stop_agent(), _evaluate_stop_convergence() (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (23): PreparedData, load_prepared_run(), PreparedData, Chargement d'un run de données préparées pour l'entraînement., Données d'un run, prêtes pour fit/predict., Résout un identifiant de run ('iris_001_001') ou un chemin complet en dossier ex, Charge train/val d'un run, sépare X/y et encode la cible en entiers.      L'enco, resolve_run_dir() (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (7): build_dataset_profile(), latest_prepared_run(), Résolution du run de données et construction du profil de dataset.  Le profil es, Renvoie l'identifiant du run le plus récent de `data/02_prepared/`.      S'appui, Construit le profil complet d'un run en un seul appel.      Combine `metadata.js, get_dataset_profile(), Renvoie en un seul appel l'ensemble des métadonnées du dataset.

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (5): agentic-ml, Conventions, graphify, Structure, État du projet

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (32): Any, ChatMistralAI, _action_signature(), agent_analyse(), agent_feature_engineering(), agent_preprocessing(), _apply_action(), make_llm() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.18
Nodes (10): Namespace, main(), parse_args(), Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr, main(), parse_args(), Entry point: run the preprocessing / feature-engineering agent system., main() (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (26): ExtraTreesClassifier, GaussianNB, HistGradientBoostingClassifier, LinearDiscriminantAnalysis, Pipeline, RandomForestClassifier, SVC, get_model_schema() (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (16): _assert_class_report(), Test de non-régression : la sortie de `launch_ml_pipeline` doit être entièrement, Vérifie la présence, la forme et la sérialisabilité du rapport par classe., test_launch_ml_pipeline_is_json_serializable(), Valide et lance la pipeline pour la config proposée, met à jour la mémoire., run_pipeline(), format_class_report(), Mise en forme lisible du rapport de classification par classe.  `val_class_repor (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (21): BaseModel, AnalysisReport, ColumnReport, GeneratedCode, Modèles Pydantic pour les sorties structurées des agents preproc.  - `AnalysisRe, Diagnostic d'une colonne., Action recommandée par l'Agent Analyse., Rapport structuré de l'Agent Analyse (structure JSON de la spec). (+13 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (22): DataSplitter, DataFrame, Path, build_preproc_graph(), Assemblage du StateGraph et point d'entrée `run_preproc_agent`.  Flux : analyse, Construit et compile le graphe de la boucle de preprocessing., Exécute la boucle de preprocessing de bout en bout et renvoie l'état final., run_preproc_agent() (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.20
Nodes (9): Ce que tu dois faire quand ce skill est invoqué, /eda, Syntaxe, Étape 1 — Parser les arguments, Étape 2 — Inspecter le dataset, Étape 3 — Calculer le nom du notebook de sortie, Étape 4 — Lire le template, Étape 5 — Générer le notebook (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (14): BaseCallbackHandler, LLMResult, make_llm(), Instancie le client Mistral avec rate limiting (clé lue dans MISTRAL_API_KEY)., get_rate_limiter(), MistralRateLimitCallback, MistralRateLimiter, Rate limiter pour l'API Mistral (thread-safe).  Fenêtres glissantes séparées : (+6 more)

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (10): _column_profile(), _correlations(), _iqr_outliers(), profile_dataframe(), Profilage d'un DataFrame en statistiques agrégées (sans données brutes).  L'Agen, Construit le profil statistique complet du DataFrame.      Args:         df: Dat, Profil d'une colonne : type, nulls, cardinalité, distribution, issues., Détection d'outliers par la méthode IQR. Renvoie ``(présence, nombre)``. (+2 more)

## Knowledge Gaps
- **11 isolated node(s):** `Syntaxe`, `Étape 1 — Parser les arguments`, `Étape 2 — Inspecter le dataset`, `Étape 3 — Calculer le nom du notebook de sortie`, `Étape 4 — Lire le template` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_estimator()` connect `Community 1` to `Community 0`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `to_search_space()` connect `Community 8` to `Community 4`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `launch_ml_pipeline()` connect `Community 7` to `Community 9`, `Community 4`, `Community 1`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **What connects `Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr`, `Entry point: run the autonomous model/hyperparameter search agent.`, `Assemblage du StateGraph et point d'entrée `run_agent`.` to the rest of the system?**
  _114 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07254623044096728 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09425287356321839 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.11092436974789915 - nodes in this community are weakly interconnected._