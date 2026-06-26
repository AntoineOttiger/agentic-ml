from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "00_raw"
PREPROC_DATA_DIR = PROJECT_ROOT / "data" / "01_preproc"
PREP_DATA_DIR = PROJECT_ROOT / "data" / "02_prepared"
DEFAULT_RAW_FILE = RAW_DATA_DIR / "creditcard.csv"

# Résultats des runs d'agent (trial_log, summary, best model)
RESULTS_DIR   = PROJECT_ROOT / "results" / "agent_runs"

TARGET_COL = "Class"
DROP_COLS  = ["Id"]

DEFAULT_MODE        = "3way"
DEFAULT_TEST_SIZE   = 0.20
DEFAULT_VAL_SIZE    = 0.20
DEFAULT_RANDOM_SEED = 42

RUN_FOLDER_WIDTH = 3

# Noms des fichiers de split produits par DataSplitter
TRAIN_FILE   = "train.csv"
VAL_FILE     = "val.csv"
TEST_FILE    = "test.csv"
PREPROC_FILE = "preprocessed.csv"

# Optimisation d'hyperparamètres (Optuna TPE)
# Bornes du garde-fou : l'agent choisit n_trials par expérience, clampé dans cet intervalle.
MIN_N_TRIALS       = 1   # plancher : assez d'essais pour que TPE soit pertinent
MAX_N_TRIALS       = 50  # plafond dur : coût Optuna / rate-limit
DEFAULT_F1_AVERAGE = "macro"  # Iris multiclasse équilibré

# Agent de recherche (LangGraph + Mistral)
AGENT_MODEL           = "mistral-small-2603"
DEFAULT_MAX_RUNS      = 10      # budget : nombre max de runs de pipeline
DEFAULT_TARGET_F1     = 0.98    # seuil cible d'arrêt sur eval_f1
CONVERGENCE_EPSILON   = 1e-3    # amélioration mini d'eval_f1 considérée significative
CONVERGENCE_PATIENCE  = 3       # K : fenêtre d'essais pour le critère de convergence
DEFAULT_STOP_MODE     = "convergence"  # "convergence" (déterministe) | "agent" (LLM décide)

# Agents preprocessing / feature engineering (LangGraph + Mistral) — système isolé
PREPROC_AGENT_MODEL     = AGENT_MODEL  # même modèle Mistral que l'agent de recherche
DEFAULT_MAX_ITERATIONS  = 10           # garde-fou de la boucle de l'orchestrateur
MAX_CODE_RETRIES        = 3            # tentatives de correction LLM sur erreur d'exécution
MAX_CATEGORICAL_UNIQUES = 50           # cardinalité au-delà de laquelle on ne passe que le compte

# Limites API Mistral
MISTRAL_RPS = 1           # requêtes par seconde (free tier)
MISTRAL_TPM = 49_000     # tokens par minute  (free tier, fenêtre 60 s)
