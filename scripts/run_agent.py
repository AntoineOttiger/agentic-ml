"""Entry point: run the autonomous model/hyperparameter search agent."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from agentic_ml.config import (
    AGENT_MODEL,
    DEFAULT_MAX_RUNS,
    DEFAULT_N_TRIALS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_F1,
)
from agentic_ml.agent import run_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the LangGraph search agent over a prepared data run."
    )
    p.add_argument(
        "--run",
        default=None,
        help="Prepared run id (e.g. 'iris_001'). Default: most recent run.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=DEFAULT_MAX_RUNS,
        help="Pipeline-run budget. Default: %(default)s",
    )
    p.add_argument(
        "--target-f1",
        type=float,
        default=DEFAULT_TARGET_F1,
        help="Stop once best eval_f1 reaches this. Default: %(default)s",
    )
    p.add_argument(
        "--model",
        default=AGENT_MODEL,
        help="Mistral model id. Default: %(default)s",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Optuna trials per pipeline run. Default: %(default)s",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed. Default: %(default)s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()

    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("MISTRAL_API_KEY is not set — export it before running the agent.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    final_state = run_agent(
        prepared_run=args.run,
        max_runs=args.max_runs,
        target_f1=args.target_f1,
        model=args.model,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    best = final_state.get("best_trial")
    print("\n=== Recherche terminée ===")
    print(f"Run de données : {final_state['prepared_run']}")
    print(f"Essais réalisés : {final_state.get('runs_used', 0)} / {args.max_runs}")
    if best:
        print(f"Meilleur modèle : {best['model_type']} (eval_f1={best['eval_f1']:.4f})")
        print(f"Hyperparamètres : {json.dumps(best['hyperparameters'], ensure_ascii=False)}")
    else:
        print("Aucun essai valide n'a été enregistré.")


if __name__ == "__main__":
    main()
