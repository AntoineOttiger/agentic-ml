from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "00_raw"
PREP_DATA_DIR = PROJECT_ROOT / "data" / "02_prepared"
RAW_IRIS_FILE = RAW_DATA_DIR / "Iris.csv"

# Résultats des runs d'agent (trial_log, summary, best model)
RESULTS_DIR   = PROJECT_ROOT / "results" / "agent_runs"

TARGET_COL = "Species"
DROP_COLS  = ["Id"]

DEFAULT_MODE        = "3way"
DEFAULT_TEST_SIZE   = 0.20
DEFAULT_VAL_SIZE    = 0.20
DEFAULT_RANDOM_SEED = 42

RUN_FOLDER_WIDTH = 3

# Noms des fichiers de split produits par DataSplitter
TRAIN_FILE = "train.csv"
VAL_FILE   = "val.csv"
TEST_FILE  = "test.csv"

# Optimisation d'hyperparamètres (Optuna TPE)
DEFAULT_N_TRIALS   = 50
DEFAULT_F1_AVERAGE = "macro"  # Iris multiclasse équilibré

# Agent de recherche (LangGraph + Mistral)
AGENT_MODEL           = "mistral-large-latest"
DEFAULT_MAX_RUNS      = 10      # budget : nombre max de runs de pipeline
DEFAULT_TARGET_F1     = 0.98    # seuil cible d'arrêt sur eval_f1
CONVERGENCE_EPSILON   = 1e-3    # amélioration mini d'eval_f1 considérée significative
CONVERGENCE_PATIENCE  = 3       # K : fenêtre d'essais pour le critère de convergence
DEFAULT_STOP_MODE     = "convergence"  # "convergence" (déterministe) | "agent" (LLM décide)
