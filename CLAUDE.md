# agentic-ml

Automatisation de la recherche de modèle ML et d'hyperparamètres via un agent IA (LangGraph/LangChain). La préparation et le nettoyage des données restent manuels.

### Skills (`.claude/skills/`)

- `agentic-archi.md` → Architecture détaillée de l'agent. **Ne jamais modifier ce fichier.**

### Structure

- `data/` → `00_raw/`dataset Iris (CSV), `01_prepared/`
- `notebooks/` → exploration
- `scripts/` → points d'entrée d'exécution (aucune logique métier)
- `src/` → code source
- `tests/` → tests

### Conventions

- Chemins et constantes centralisés dans `src/agentic_ml/config.py`
