"""Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run préparé."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Sortie console en UTF-8 (les terminaux Windows utilisent cp1252 par défaut).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agentic_ml.config import (
    DEFAULT_F1_AVERAGE,
    DEFAULT_N_TRIALS,
    DEFAULT_RANDOM_SEED,
)
from agentic_ml.training import HyperparameterOptimizer, available_models


#python scripts/run_optimization.py --run iris_001 --model random_forest --search-space configs/search_spaces/random_forest.json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimise les hyperparamètres d'un modèle par Tree-structured "
        "Parzen Estimator (Optuna) sur un run de données préparées."
    )
    p.add_argument(
        "--run",
        required=True,
        help="Identifiant du run préparé (ex. 'iris_001_001') ou chemin vers le dossier.",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=available_models(),
        help="Modèle à optimiser.",
    )
    p.add_argument(
        "--search-space",
        required=True,
        type=Path,
        help="Chemin du fichier JSON décrivant les plages d'hyperparamètres.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Nombre d'essais Optuna. Default: %(default)s",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Graine aléatoire pour la reproductibilité. Default: %(default)s",
    )
    p.add_argument(
        "--f1-average",
        default=DEFAULT_F1_AVERAGE,
        help="Stratégie de moyennage du F1-score. Default: %(default)s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    search_space = json.loads(args.search_space.read_text(encoding="utf-8"))

    optimizer = HyperparameterOptimizer(
        f1_average=args.f1_average,
        random_seed=args.seed,
        n_trials=args.n_trials,
    )
    result = optimizer.optimize(args.run, args.model, search_space)

    print(f"Modèle           : {result.model_type}")
    print(f"Essais           : {result.n_trials}")
    print(f"F1 train         : {result.train_f1:.4f}")
    print(f"F1 eval (val)    : {result.eval_f1:.4f}")
    print(f"Overfitting (Δ)  : {result.train_f1 - result.eval_f1:+.4f}")
    print("Meilleurs hyperparamètres :")
    print(json.dumps(result.best_params, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
