# Graph Report - agentic-ml  (2026-06-27)

## Corpus Check
- 31 files · ~8,355 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 214 nodes · 321 edges · 20 communities (16 shown, 4 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `56d91c13`
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
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `AgentState` - 14 edges
2. `run_agent()` - 10 edges
3. `launch_ml_pipeline()` - 9 edges
4. `HyperparameterOptimizer` - 9 edges
5. `available_models()` - 8 edges
6. `RateLimiter` - 8 edges
7. `_save_best_model()` - 7 edges
8. `persist_results()` - 7 edges
9. `DataSplitter` - 7 edges
10. `load_prepared_run()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `parse_args()` --calls--> `available_models()`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/models.py
- `main()` --calls--> `HyperparameterOptimizer`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/optimizer.py
- `main()` --calls--> `run_agent()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/graph.py
- `main()` --calls--> `format_class_report()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/report.py
- `test_launch_ml_pipeline_is_json_serializable()` --calls--> `launch_ml_pipeline()`  [INFERRED]
  tests/test_pipeline_serialization.py → src/agentic_ml/agents/training_agent/tools.py

## Import Cycles
- None detected.

## Communities (20 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (29): BaseChatModel, Path, build_agent_graph(), Construit et compile le graphe de la boucle de recherche.      Flux : propose_ex, Exécute la boucle agentique de bout en bout et renvoie l'état final.      Args:, run_agent(), evaluate_stop(), _evaluate_stop_agent() (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (26): État partagé de l'agent, réinjecté à chaque tour de la boucle LangGraph.  Reflèt, Un essai enregistré dans le `trial_log`., Trial, load_prepared_run(), PreparedData, Chargement d'un run de données préparées pour l'entraînement., Données d'un run, prêtes pour fit/predict., Résout un identifiant de run ('iris_001_001') ou un chemin complet en dossier ex (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (22): Any, build_dataset_profile(), Construit le profil complet d'un run en un seul appel.      Combine `metadata.js, build_context(), _model_schemas(), System prompt (mission) et construction du contexte injecté à chaque tour., Schémas des hyperparamètres valides de tous les modèles disponibles., Assemble le contexte dynamique pour le nœud propose_experiment. (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (5): agentic-ml, Conventions, graphify, Structure, État du projet

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (14): Namespace, main(), parse_args(), Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr, main(), parse_args(), Entry point: run the preprocessing / feature-engineering agent system., main() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (21): ExtraTreesClassifier, GaussianNB, HistGradientBoostingClassifier, LinearDiscriminantAnalysis, Pipeline, RandomForestClassifier, SVC, _build_extra_trees() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.50
Nodes (4): _assert_class_report(), Test de non-régression : la sortie de `launch_ml_pipeline` doit être entièrement, Vérifie la présence, la forme et la sérialisabilité du rapport par classe., test_launch_ml_pipeline_is_json_serializable()

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (10): BaseModel, Experiment, HyperparameterRange, Sortie structurée du LLM pour une proposition d'essai.  Le LLM renvoie une liste, Plage d'un hyperparamètre à explorer., Proposition d'essai : hypothèse explicite + configuration., Décision d'arrêt prise par le LLM (mode d'arrêt « agent »)., Convertit la liste de plages en dict au format `search_space`. (+2 more)

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (7): DataSplitter, DataFrame, _latest_preproc_run(), main(), parse_args(), Entry point: split a preprocessed dataset into train/val/test partitions., _resolve_preproc_run()

### Community 10 - "Community 10"
Cohesion: 0.20
Nodes (9): Ce que tu dois faire quand ce skill est invoqué, /eda, Syntaxe, Étape 1 — Parser les arguments, Étape 2 — Inspecter le dataset, Étape 3 — Calculer le nom du notebook de sortie, Étape 4 — Lire le template, Étape 5 — Générer le notebook (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (5): Valide et lance la pipeline pour la config proposée, met à jour la mémoire., run_pipeline(), format_class_report(), Mise en forme lisible du rapport de classification par classe.  `val_class_repor, Rend le rapport par classe sous forme de tableau aligné multi-lignes.

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (12): BaseCallbackHandler, LLMResult, get_rate_limiter(), RateLimitCallback, RateLimiter, Rate limiter pour les APIs LLM (thread-safe).  Fenêtres glissantes séparées :, Retourne (ou crée) le singleton RateLimiter partagé par tous les agents., Sliding-window rate limiter pour RPS et TPM.      Thread-safe : le lock n'est te (+4 more)

## Knowledge Gaps
- **11 isolated node(s):** `Syntaxe`, `Étape 1 — Parser les arguments`, `Étape 2 — Inspecter le dataset`, `Étape 3 — Calculer le nom du notebook de sortie`, `Étape 4 — Lire le template` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_estimator()` connect `Community 1` to `Community 0`, `Community 2`, `Community 6`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `AgentState` connect `Community 0` to `Community 1`, `Community 2`, `Community 12`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `launch_ml_pipeline()` connect `Community 2` to `Community 0`, `Community 1`, `Community 12`, `Community 7`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `run_agent()` (e.g. with `main()` and `make_llm()`) actually correct?**
  _`run_agent()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `launch_ml_pipeline()` (e.g. with `test_launch_ml_pipeline_is_json_serializable()` and `run_pipeline()`) actually correct?**
  _`launch_ml_pipeline()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Entry point: split a preprocessed dataset into train/val/test partitions.`, `Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr`, `Entry point: run the preprocessing / feature-engineering agent system.` to the rest of the system?**
  _79 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1053763440860215 - nodes in this community are weakly interconnected._