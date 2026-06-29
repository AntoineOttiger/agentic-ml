from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "00_raw"
PREP_DATA_DIR = PROJECT_ROOT / "data" / "01_prepared"
DEFAULT_RAW_FILE = RAW_DATA_DIR / "iris.csv"

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
TRAIN_FILE   = "train.csv"
VAL_FILE     = "val.csv"
TEST_FILE    = "test.csv"

# Optimisation d'hyperparamètres (Optuna TPE)
# Bornes du garde-fou : l'agent choisit n_trials par expérience, clampé dans cet intervalle.
MIN_N_TRIALS       = 1   # plancher : assez d'essais pour que TPE soit pertinent
MAX_N_TRIALS       = 50  # plafond dur : coût Optuna / rate-limit
DEFAULT_F1_AVERAGE = "macro"  # Iris multiclasse équilibré

# Provider LLM — unique point de configuration pour tous les agents
AGENT_PROVIDER = "mistral"  # "mistral" | "anthropic"

MISTRAL_AGENT_MODEL   = "mistral-small-2603"
ANTHROPIC_AGENT_MODEL = "claude-haiku-4-5-20251001"

# Modèle actif dérivé du provider
AGENT_MODEL        = MISTRAL_AGENT_MODEL if AGENT_PROVIDER == "mistral" else ANTHROPIC_AGENT_MODEL

DEFAULT_MAX_RUNS      = 10      # budget : nombre max de runs de pipeline
DEFAULT_TARGET_F1     = 0.98    # seuil cible d'arrêt sur eval_f1
CONVERGENCE_EPSILON   = 1e-3    # amélioration mini d'eval_f1 considérée significative
CONVERGENCE_PATIENCE  = 3       # K : fenêtre d'essais pour le critère de convergence
DEFAULT_STOP_MODE     = "convergence"  # "convergence" (déterministe) | "agent" (LLM décide)

# --- Préparation agentique des données (agents/prepare_agent) ---
# Tâches supportées pour l'instant : classification uniquement.
PREPARE_TASK_TYPES = ("classification",)

# Gate d'invariance par ligne (§6). Le test tourne sur un échantillon borné, pas
# sur les N lignes : son coût propre est constant.
GATE_SAMPLE_SIZE = 5_000           # taille max de l'échantillon E
GATE_SUBSAMPLES = ((0.4, 101), (0.7, 202))  # (fraction, seed) des sous-échantillons S₁, S₂
GATE_FLOAT_ATOL = 0.0              # tolérance absolue flottants (0 = égalité stricte)
GATE_FLOAT_RTOL = 0.0              # tolérance relative flottants
GATE_ENABLE_PERMUTATION = False    # test de permutation (§6.2) — features ordonnées/temporelles
GATE_PERMUTATION_SEED = 303
GATE_SAMPLE_SEED = 7               # seed du tirage de l'échantillon E

# Budget des boucles agentiques (garde-fous durs ; l'agent peut finir avant)
MAX_CLEANING_STEPS = 8             # transformations max appliquées en phase cleaning
MAX_FE_STEPS = 8                   # transformations max appliquées en phase feature engineering
MAX_PHASE_ATTEMPTS = 24            # itérations max par phase (réussies + rejetées par le gate)

# Limites API Mistral (free tier)
MISTRAL_RPS = 1
MISTRAL_TPM = 49_000

# Limites API Anthropic (tier 1)
ANTHROPIC_RPM = 50
ANTHROPIC_TPM = 40_000
