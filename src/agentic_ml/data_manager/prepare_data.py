from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from agentic_ml.config import (
    DEFAULT_VAL_SIZE,
    DEFAULT_MODE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    DROP_COLS,
    PREP_DATA_DIR,
    PROJECT_ROOT,
    RAW_IRIS_FILE,
    RUN_FOLDER_WIDTH,
    TARGET_COL,
)


class DataSplitter:
    def __init__(
        self,
        source_path: Path | str = RAW_IRIS_FILE,
        output_dir: Path | str = PREP_DATA_DIR,
        mode: str = DEFAULT_MODE,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        if mode not in ("2way", "3way"):
            raise ValueError(f"mode must be '2way' or '3way', got '{mode}'")

    def _next_run_folder(self) -> Path:
        dataset_name = self.source_path.stem.lower()
        prefix = f"{dataset_name}_"
        existing = [
            int(p.name[len(prefix):])
            for p in self.output_dir.iterdir()
            if p.is_dir()
            and p.name.startswith(prefix)
            and p.name[len(prefix):].isdigit()
        ]
        next_idx = max(existing, default=0) + 1
        return self.output_dir / f"{prefix}{str(next_idx).zfill(RUN_FOLDER_WIDTH)}"

    def _stratified_split(
        self, df: pd.DataFrame, frac: float, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        parts = [
            group.sample(frac=frac, random_state=seed)
            for _, group in df.groupby(TARGET_COL)
        ]
        sampled = pd.concat(parts)
        remainder = df.drop(sampled.index)
        return sampled.reset_index(drop=True), remainder.reset_index(drop=True)

    def _split_info(self, df: pd.DataFrame, filename: str) -> dict:
        return {
            "file": filename,
            "n_rows": len(df),
            "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        }

    def split_and_save(self) -> Path:
        df = pd.read_csv(self.source_path)

        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COL}' not found in {self.source_path}. "
                f"Available columns: {list(df.columns)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_folder = self._next_run_folder()
        run_folder.mkdir()

        test_df, remainder = self._stratified_split(df, self.test_size, self.random_seed)

        splits: dict[str, pd.DataFrame] = {}
        if self.mode == "3way":
            adjusted_val_frac = self.val_size / (1.0 - self.test_size)
            val_df, train_df = self._stratified_split(
                remainder, adjusted_val_frac, self.random_seed
            )
            splits = {"train": train_df, "val": val_df, "test": test_df}
        else:
            splits = {"train": remainder, "test": test_df}

        for name, split_df in splits.items():
            split_df.to_csv(run_folder / f"{name}.csv", index=False)

        feature_cols = [c for c in df.columns if c != TARGET_COL]
        run_id = run_folder.name
        metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": str(self.source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mode": self.mode,
            "random_seed": self.random_seed,
            "target_column": TARGET_COL,
            "feature_columns": feature_cols,
            "splits": {
                name: self._split_info(split_df, f"{name}.csv")
                for name, split_df in splits.items()
            },
            "split_ratios": {
                "test_size": self.test_size,
                **({"val_size": self.val_size} if self.mode == "3way" else {}),
            },
        }
        (run_folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        return run_folder
