# Graph Report - .  (2026-06-25)

## Corpus Check
- Corpus is ~1,778 words - fits in a single context window. You may not need a graph.

## Summary
- 38 nodes · 44 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.77)
- Token cost: 28,105 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Agent Search Loop (LangGraph)|Agent Search Loop (LangGraph)]]
- [[_COMMUNITY_Data & Optimization Layer|Data & Optimization Layer]]
- [[_COMMUNITY_Agent Entry Point|Agent Entry Point]]
- [[_COMMUNITY_ML Pipeline Tools|ML Pipeline Tools]]
- [[_COMMUNITY_Data Preparation CLI|Data Preparation CLI]]
- [[_COMMUNITY_Optimization CLI|Optimization CLI]]
- [[_COMMUNITY_Agent Mission & Objective|Agent Mission & Objective]]
- [[_COMMUNITY_Dataset Profiling Tool|Dataset Profiling Tool]]

## God Nodes (most connected - your core abstractions)
1. `Boucle de recherche dirigée de l'agent ML` - 7 edges
2. `Nœud propose_experiment` - 6 edges
3. `Nœud run_pipeline` - 5 edges
4. `Nœud evaluate_stop` - 5 edges
5. `Tool launch_ml_pipeline` - 4 edges
6. `parse_args()` - 3 edges
7. `parse_args()` - 3 edges
8. `parse_args()` - 3 edges
9. `État de l'agent (state / trial_log)` - 3 edges
10. `Tool get_model_schema` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Boucle de recherche dirigée de l'agent ML` --references--> `langgraph (dépendance)`  [INFERRED]
  .claude/skills/agentic-archi.md → requirements.txt
- `Projet agentic-ml` --references--> `Boucle de recherche dirigée de l'agent ML`  [INFERRED]
  CLAUDE.md → .claude/skills/agentic-archi.md
- `Tool launch_ml_pipeline` --calls--> `HyperparameterOptimizer (module training)`  [INFERRED]
  .claude/skills/agentic-archi.md → CLAUDE.md
- `Mission (system prompt de l'agent ML)` --references--> `langchain-mistralai (dépendance)`  [INFERRED]
  .claude/skills/agentic-archi.md → requirements.txt
- `Tool launch_ml_pipeline` --references--> `xgboost (dépendance)`  [INFERRED]
  .claude/skills/agentic-archi.md → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Boucle agentique LangGraph (3 nœuds)** — agentic_archi_propose_experiment, agentic_archi_run_pipeline, agentic_archi_evaluate_stop [EXTRACTED 1.00]
- **Tools de l'agent ML** — agentic_archi_get_dataset_profile, agentic_archi_get_model_schema, agentic_archi_query_ml_knowledge_base, agentic_archi_launch_ml_pipeline [EXTRACTED 1.00]
- **Critères d'arrêt chiffrés** — agentic_archi_stop_criteria, agentic_archi_evaluate_stop, agentic_archi_objective_eval_f1 [EXTRACTED 1.00]

## Communities (9 total, 1 thin omitted)

### Community 0 - "Agent Search Loop (LangGraph)"
Cohesion: 0.36
Nodes (9): Boucle de recherche dirigée de l'agent ML, État de l'agent (state / trial_log), Nœud evaluate_stop, Observabilité (trace / rejeu de trajectoire), Signal d'overfitting (train_f1 - eval_f1), Nœud propose_experiment, Nœud run_pipeline, Critères d'arrêt (budget/convergence/seuil) (+1 more)

### Community 1 - "Data & Optimization Layer"
Cohesion: 0.33
Nodes (6): config.py (chemins et constantes centralisés), DataSplitter (couche données), HyperparameterOptimizer (module training), Projet agentic-ml, optuna (dépendance), scikit-learn (dépendance)

### Community 2 - "Agent Entry Point"
Cohesion: 0.50
Nodes (4): Namespace, main(), parse_args(), Entry point: run the autonomous model/hyperparameter search agent.

### Community 3 - "ML Pipeline Tools"
Cohesion: 0.50
Nodes (4): Tool get_model_schema, Tool launch_ml_pipeline, Tool query_ml_knowledge_base (RAG), xgboost (dépendance)

### Community 4 - "Data Preparation CLI"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Entry point: split the Iris dataset into train/val/test partitions.

### Community 5 - "Optimization CLI"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr

### Community 6 - "Agent Mission & Objective"
Cohesion: 0.67
Nodes (3): Objectif: maximiser eval_f1, Mission (system prompt de l'agent ML), langchain-mistralai (dépendance)

## Knowledge Gaps
- **9 isolated node(s):** `dataset_profile (contexte initial)`, `Tool get_dataset_profile`, `Tool query_ml_knowledge_base (RAG)`, `config.py (chemins et constantes centralisés)`, `langgraph (dépendance)` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Boucle de recherche dirigée de l'agent ML` connect `Agent Search Loop (LangGraph)` to `Data & Optimization Layer`, `Agent Mission & Objective`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `Projet agentic-ml` connect `Data & Optimization Layer` to `Agent Search Loop (LangGraph)`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `Nœud propose_experiment` connect `Agent Search Loop (LangGraph)` to `ML Pipeline Tools`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Boucle de recherche dirigée de l'agent ML` (e.g. with `langgraph (dépendance)` and `Projet agentic-ml`) actually correct?**
  _`Boucle de recherche dirigée de l'agent ML` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Tool launch_ml_pipeline` (e.g. with `HyperparameterOptimizer (module training)` and `xgboost (dépendance)`) actually correct?**
  _`Tool launch_ml_pipeline` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Entry point: split the Iris dataset into train/val/test partitions.`, `Entry point: run the autonomous model/hyperparameter search agent.`, `Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._