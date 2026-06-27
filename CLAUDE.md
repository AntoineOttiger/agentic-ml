# agentic-ml

Automatisation de la recherche de modèle ML et d'hyperparamètres via un agent IA (LangGraph/LangChain). La préparation et le nettoyage des données restent manuels.

### Structure

- `data/00_raw/` → dataset Iris (CSV)
- `data/01_prepared/` → splits versionnés `<dataset>_<prepared_idx>` (`iris_001`, …) — chaque run contient `train.csv`, `val.csv`, `test.csv`, `metadata.json`
- `notebooks/` → exploration
- `src/agentic_ml/data/prepare_data.py` → `DataSplitter` : split stratifié 2way ou 3way
- `src/agentic_ml/training`→ module de gestion taining/eval/optim des models , HyperparameterOptimizer à utiliser par l'agent
- `src/agentic_ml/agent` → module de gestion de l'agent
- `src/agentic_ml/config.py` → chemins et constantes centralisés
- `tests/` → vide (agent non encore implémenté)
- `results/agent_runs` → resulat

### Conventions

- Tous les chemins et constantes passent par `config.py`.
- Les splits sont générés via `python scripts/prepare_data.py [--mode 2way|3way] [--test-size] [--val-size] [--seed]`.
- Défauts : mode `3way`, test 20 %, val 20 %, seed 42.

### État du projet

La couche données (`DataSplitter`) est implémentée. L'agent (nœuds LangGraph, tools ML, pipeline) reste à créer.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
