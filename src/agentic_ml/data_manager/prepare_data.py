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
    RUN_FOLDER_WIDTH,
    TARGET_COL,
)


def next_run_folder(
    source_file: Path | str, output_dir: Path | str = PREP_DATA_DIR
) -> Path:
    """Renvoie le prochain dossier de run versionné `<dataset>_<idx>` (idx zfill).

    Source de vérité unique pour la convention de nommage des splits, partagée
    entre `DataSplitter` et le `prepare_agent` (qui matérialise ses artefacts au
    même endroit). Crée `output_dir` si besoin mais n'alloue pas le sous-dossier.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{Path(source_file).stem.lower()}_"
    existing = [
        int(p.name[len(prefix):])
        for p in output_dir.iterdir()
        if p.is_dir()
        and p.name.startswith(prefix)
        and p.name[len(prefix):].isdigit()
    ]
    next_idx = max(existing, default=0) + 1
    return output_dir / f"{prefix}{str(next_idx).zfill(RUN_FOLDER_WIDTH)}"


class DataSplitter:
    def __init__(
        self,
        source_file: Path | str,
        output_dir: Path | str = PREP_DATA_DIR,
        mode: str = DEFAULT_MODE,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        if mode not in ("2way", "3way"):
            raise ValueError(f"mode must be '2way' or '3way', got '{mode}'")

    def _next_run_folder(self) -> Path:
        return next_run_folder(self.source_file, self.output_dir)

    def _stratified_split(
        self, df: pd.DataFrame, frac: float, seed: int, target_col: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        target = target_col or TARGET_COL
        parts = [
            group.sample(frac=frac, random_state=seed)
            for _, group in df.groupby(target)
        ]
        sampled = pd.concat(parts)
        remainder = df.drop(sampled.index)
        return sampled.reset_index(drop=True), remainder.reset_index(drop=True)

    def split_frame(
        self, df: pd.DataFrame, target_column: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """Partitionne un dataframe **déjà en mémoire** (classification stratifiée).

        Utilisé par la phase 3 du `prepare_agent`, qui possède déjà le dataframe
        post-feature-engineering : on ne relit pas de CSV et on ne supprime aucune
        colonne ici (le nettoyage des colonnes relève de la phase 1). Renvoie les
        partitions sans les écrire — l'écriture des artefacts est gérée par l'appelant.
        """
        target = target_column or TARGET_COL
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. Available columns: {list(df.columns)}"
            )

        test_df, remainder = self._stratified_split(
            df, self.test_size, self.random_seed, target
        )
        if self.mode == "3way":
            adjusted_val_frac = self.val_size / (1.0 - self.test_size)
            val_df, train_df = self._stratified_split(
                remainder, adjusted_val_frac, self.random_seed, target
            )
            return {"train": train_df, "val": val_df, "test": test_df}
        return {"train": remainder, "test": test_df}

    def _split_info(self, df: pd.DataFrame, filename: str) -> dict:
        return {
            "file": filename,
            "n_rows": len(df),
            "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        }

    def split_and_save(self) -> Path:
        df = pd.read_csv(self.source_file)

        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COL}' not found in {self.source_file}. "
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
            "source_file": str(self.source_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
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
