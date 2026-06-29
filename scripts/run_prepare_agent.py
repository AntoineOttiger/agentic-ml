"""Entry point: run the agentic data-preparation pipeline (cleaning → FE → split)."""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from agentic_ml.config import (
    AGENT_MODEL,
    AGENT_PROVIDER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RAW_FILE,
    TARGET_COL,
)
from agentic_ml.agents.prepare_agent import run_prepare_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the agentic data-preparation pipeline over a raw CSV.",
    )
    p.add_argument(
        "--input-csv",
        default=str(DEFAULT_RAW_FILE),
        help="Path to the raw .csv file. Default: %(default)s",
    )
    p.add_argument(
        "--target-column",
        default=TARGET_COL,
        help="Name of the target column (never transformed). Default: %(default)s",
    )
    p.add_argument(
        "--task-type",
        default="classification",
        choices=["classification"],
        help="Task type. Default: %(default)s",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed. Default: %(default)s"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Produce the recipe without materialising the splits (leakage audit).",
    )
    p.add_argument(
        "--enable-permutation",
        action="store_true",
        help="Enable the §6.2 permutation check (ordered/temporal features).",
    )
    p.add_argument("--model", default=AGENT_MODEL, help="LLM model id. Default: %(default)s")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    key_name = "ANTHROPIC_API_KEY" if AGENT_PROVIDER == "anthropic" else "MISTRAL_API_KEY"
    if not os.environ.get(key_name):
        sys.exit(f"{key_name} is not set — export it before running the agent.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    final = run_prepare_agent(
        input_csv=args.input_csv,
        target_column=args.target_column,
        task_type=args.task_type,
        seed=args.seed,
        dry_run=args.dry_run,
        enable_permutation=args.enable_permutation,
        model=args.model,
    )

    print("\n=== Préparation terminée ===")
    print(f"Transformations cleaning : {final.cleaning_steps}")
    print(f"Features ajoutées (FE)   : {final.fe_steps}")
    print(f"Artefacts                : {final.output_path}")
    if final.df is not None:
        print(f"Schéma final             : {final.df.shape[1]} colonnes")


if __name__ == "__main__":
    main()
