"""Entry point: split the Iris dataset into train/val/test partitions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_ml.config import (
    DEFAULT_VAL_SIZE,
    DEFAULT_MODE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
)
from agentic_ml.data.prepare_data import DataSplitter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split the Iris dataset into stratified train/val/test partitions."
    )
    p.add_argument(
        "--mode",
        choices=["2way", "3way"],
        default=DEFAULT_MODE,
        help="'2way' (train+test) or '3way' (train+val+test). Default: %(default)s",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of total data reserved for test. Default: %(default)s",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Fraction of total data reserved for val (3way only). Default: %(default)s",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility. Default: %(default)s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splitter = DataSplitter(
        mode=args.mode,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )
    run_folder = splitter.split_and_save()
    print(f"Splits saved to: {run_folder}")


if __name__ == "__main__":
    main()
