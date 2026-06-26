"""Entry point: run the preprocessing / feature-engineering agent system."""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from agentic_ml.config import (
    DEFAULT_MAX_ITERATIONS,
    PREPROC_AGENT_MODEL,
    RAW_IRIS_FILE,
    TARGET_COL,
)
from agentic_ml.agents.preproc_agent import run_preproc_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the preprocessing/feature-engineering agent over a raw CSV."
    )
    p.add_argument(
        "--input",
        default=str(RAW_IRIS_FILE),
        help="Input CSV path. Default: %(default)s",
    )
    p.add_argument(
        "--target",
        default=TARGET_COL,
        help="Target column name. Default: %(default)s",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Orchestrator loop safety cap. Default: %(default)s",
    )
    p.add_argument(
        "--model",
        default=PREPROC_AGENT_MODEL,
        help="Mistral model id. Default: %(default)s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()

    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("MISTRAL_API_KEY is not set — export it before running the agent.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    final_state = run_preproc_agent(
        input_csv=args.input,
        target_column=args.target,
        max_iterations=args.max_iterations,
        model=args.model,
    )

    print("\n=== Preprocessing terminé ===")
    print(f"Itérations : {final_state.get('iteration_count', 0)} / {args.max_iterations}")
    applied = final_state.get("applied_transformations", [])
    created = final_state.get("created_features", [])
    print(f"Transformations appliquées ({len(applied)}) : {applied}")
    print(f"Features créées ({len(created)}) : {created}")
    df = final_state.get("dataframe")
    if df is not None:
        print(f"DataFrame final : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"Colonnes : {list(df.columns)}")


if __name__ == "__main__":
    main()
