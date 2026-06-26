# Graph Report - agentic-ml  (2026-06-26)

## Corpus Check
- 42 files · ~10,063 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 263 nodes · 416 edges · 20 communities (14 shown, 6 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cff3bcf5`
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
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]

## God Nodes (most connected - your core abstractions)
1. `MLPipelineState` - 14 edges
2. `AgentState` - 14 edges
3. `MCPToolClient` - 13 edges
4. `_apply_action()` - 9 edges
5. `HyperparameterOptimizer` - 9 edges
6. `agent_preprocessing()` - 8 edges
7. `agent_feature_engineering()` - 8 edges
8. `run_agent()` - 8 edges
9. `available_models()` - 8 edges
10. `agent_analyse()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `DataSplitter`  [INFERRED]
  scripts/prepare_data.py → src/agentic_ml/data_manager/prepare_data.py
- `main()` --calls--> `run_agent()`  [INFERRED]
  scripts/run_agent.py → src/agentic_ml/agents/training_agent/graph.py
- `parse_args()` --calls--> `available_models()`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/models.py
- `main()` --calls--> `HyperparameterOptimizer`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/optimizer.py
- `main()` --calls--> `run_preproc_agent()`  [INFERRED]
  scripts/run_preproc_agent.py → src/agentic_ml/agents/preproc_agent/graph.py

## Import Cycles
- None detected.

## Communities (20 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (15): MCPToolClient, Client MCP synchrone pour les outils ML de l'agent.  Wrapping synchrone du clien, Context manager synchrone autour d'une session MCP stdio.      Usage:         wi, Coroutine principal : ouvre la connexion, signale 'ready', attend 'stop'., Appelle un outil MCP et retourne le résultat désérialisé.          Timeout de 30, build_agent_graph(), Assemblage du StateGraph et point d'entrée `run_agent`., Construit et compile le graphe de la boucle de recherche.      Flux : propose_ex (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (20): load_prepared_run(), PreparedData, Chargement d'un run de données préparées pour l'entraînement., Données d'un run, prêtes pour fit/predict., Résout un identifiant de run ('001') ou un chemin complet en dossier existant., Charge train/val d'un run, sépare X/y et encode la cible en entiers.      L'enco, resolve_run_dir(), build_estimator() (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (14): get_dataset_profile_tool(), get_model_schema_tool(), launch_ml_pipeline_tool(), Serveur MCP exposant les 3 outils ML via transport stdio., Retourne les métadonnées complètes du dataset pour un run préparé donné., Retourne le schéma typé des hyperparamètres valides pour un type de modèle., Lance l'optimisation Optuna + entraînement pour une configuration donnée.      R, build_dataset_profile() (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (5): agentic-ml, Conventions, graphify, Structure, État du projet

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (36): Any, ChatMistralAI, _action_signature(), agent_analyse(), agent_feature_engineering(), agent_preprocessing(), _apply_action(), make_llm() (+28 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (18): Namespace, build_preproc_graph(), Assemblage du StateGraph et point d'entrée `run_preproc_agent`.  Flux : analyse, Construit et compile le graphe de la boucle de preprocessing., Exécute la boucle de preprocessing de bout en bout et renvoie l'état final., run_preproc_agent(), main(), parse_args() (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (31): ExtraTreesClassifier, GaussianNB, HistGradientBoostingClassifier, LinearDiscriminantAnalysis, Pipeline, RandomForestClassifier, SVC, get_model_schema() (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (27): evaluate_stop(), _evaluate_stop_agent(), _evaluate_stop_convergence(), make_llm(), Nœuds de la boucle agentique.  Seul `propose_experiment` mobilise le LLM. `run_p, Décide de continuer ou d'arrêter, selon le `stop_mode`.      Le budget (`max_run, Critères d'arrêt déterministes : seuil cible puis convergence., Délègue la décision d'arrêt au LLM (budget déjà borné en amont). (+19 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (21): BaseModel, AnalysisReport, ColumnReport, GeneratedCode, Modèles Pydantic pour les sorties structurées des agents preproc.  - `AnalysisRe, Diagnostic d'une colonne., Action recommandée par l'Agent Analyse., Rapport structuré de l'Agent Analyse (structure JSON de la spec). (+13 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (12): DataSplitter, DataFrame, Path, _next_run_folder(), persist_results(), Persistance d'une run de preprocessing dans `data/01_preproc/<dataset>_NNN/`.  É, Renvoie le prochain dossier `<dataset>_NNN` disponible., Écrit les artefacts de la run et renvoie le dossier créé. (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.38
Nodes (6): _column_profile(), _iqr_outliers(), Profilage d'un DataFrame en statistiques agrégées (sans données brutes).  L'Agen, Profil d'une colonne : type, nulls, cardinalité, distribution, issues., Détection d'outliers par la méthode IQR. Renvoie ``(présence, nombre)``., Series

## Knowledge Gaps
- **4 isolated node(s):** `Structure`, `Conventions`, `État du projet`, `graphify`
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `to_search_space()` connect `Community 8` to `Community 4`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `build_estimator()` connect `Community 1` to `Community 4`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **What connects `Entry point: split the Iris dataset into train/val/test partitions.`, `Entry point: run the autonomous model/hyperparameter search agent.`, `Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr` to the rest of the system?**
  _105 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1067193675889328 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10541310541310542 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.09716599190283401 - nodes in this community are weakly interconnected._