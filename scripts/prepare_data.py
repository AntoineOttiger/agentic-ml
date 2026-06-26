"""Entry point: split a preprocessed dataset into train/val/test partitions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_ml.config import (
    DEFAULT_VAL_SIZE,
    DEFAULT_MODE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    PREPROC_DATA_DIR,
    RUN_FOLDER_WIDTH,
)
from agentic_ml.data_manager.prepare_data import DataSplitter


def _latest_preproc_run() -> Path:
    candidates: list[tuple[int, Path]] = []
    if PREPROC_DATA_DIR.is_dir():
        for p in PREPROC_DATA_DIR.iterdir():
            if not p.is_dir() or "_" not in p.name:
                continue
            suffix = p.name.rsplit("_", 1)[1]
            if suffix.isdigit():
                candidates.append((int(suffix), p))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun run préprocessé dans {PREPROC_DATA_DIR}. "
            "Lancez d'abord le preproc_agent."
        )
    return max(candidates)[1]


def _resolve_preproc_run(value: str | None) -> Path:
    if value is None:
        return _latest_preproc_run()
    p = Path(value)
    if p.is_absolute():
        return p
    if (PREPROC_DATA_DIR / p).is_dir():
        return PREPROC_DATA_DIR / p
    raise FileNotFoundError(f"Run preproc introuvable : {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a preprocessed dataset into stratified train/val/test partitions."
    )
    p.add_argument(
        "--preproc-run",
        default=None,
        metavar="RUN_ID",
        help="ID du run preproc à utiliser (ex. iris_001). "
             "Par défaut : dernier run détecté dans data/01_preproc/.",
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
    preproc_run_dir = _resolve_preproc_run(args.preproc_run)
    splitter = DataSplitter(
        preproc_run_dir=preproc_run_dir,
        mode=args.mode,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )
    run_folder = splitter.split_and_save()
    print(f"Splits saved to: {run_folder}")


if __name__ == "__main__":
    main()
