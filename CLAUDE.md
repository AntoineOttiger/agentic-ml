# agentic-ml

Automatisation de la recherche de modèle ML et d'hyperparamètres via un agent IA (LangGraph/LangChain). La préparation et le nettoyage des données restent manuels.

### Skills (`.claude/skills/`)

- `agentic-archi.md` → Architecture détaillée de l'agent. **Ne jamais modifier ce fichier.**

### Structure

- `data/00_raw/` → dataset Iris (CSV)
- `data/01_prepared/` → splits versionnés (`001/`, `002/`, …) — chaque run contient `train.csv`, `val.csv`, `test.csv`, `metadata.json`
- `notebooks/` → exploration
- `scripts/prepare_data.py` → CLI pour `DataSplitter` (aucune logique métier)
- `src/agentic_ml/data/prepare_data.py` → `DataSplitter` : split stratifié 2way ou 3way
- `src/agentic_ml/training`
- `src/agentic_ml/config.py` → chemins et constantes centralisés
- `tests/` → vide (agent non encore implémenté)

### Conventions

- Tous les chemins et constantes passent par `config.py`.
- Les splits sont générés via `python scripts/prepare_data.py [--mode 2way|3way] [--test-size] [--val-size] [--seed]`.
- Défauts : mode `3way`, test 20 %, val 20 %, seed 42.

### État du projet

La couche données (`DataSplitter`) est implémentée. L'agent (nœuds LangGraph, tools ML, pipeline) reste à créer.
