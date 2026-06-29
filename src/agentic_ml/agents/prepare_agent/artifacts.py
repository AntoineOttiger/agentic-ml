"""Écriture des artefacts de sortie (§10) dans `output_dir`.

- train/val/test.csv (+ dtypes.json pour réimposer les types au rechargement)
- transformations.json + transforms/*.py (recette rejouable, déléguée à recipe.py)
- metadata.json (seed, ratios, hash source, versions, schéma, cible, task_type)
- schema.json (schéma final)

En mode dry-run, on n'écrit que la recette et un audit : aucune partition n'est
matérialisée (§11 : « produit la recette sans muter les données »).
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from agentic_ml.config import DEFAULT_MODE, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE
from agentic_ml.agents.prepare_agent.recipe import write_recipe
from agentic_ml.agents.prepare_agent.state import PrepState


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _lib_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("pandas", "numpy", "pydantic", "langgraph", "scikit-learn"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _dtypes_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c): str(dt) for c, dt in df.dtypes.items()}


def write_artifacts(state: PrepState) -> Path:
    """Matérialise tous les artefacts et renvoie le répertoire de sortie."""
    cfg = state.config
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recette rejouable (toujours écrite, y compris en dry-run pour l'audit).
    write_recipe(state.history, output_dir)

    df = state.df
    final_schema = [c.model_dump() for c in state.current_schema]
    _write_json(output_dir / "schema.json", final_schema)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(Path(cfg.input_csv).as_posix()),
        "source_hash": state.source_hash,
        "target_column": cfg.target_column,
        "task_type": cfg.task_type,
        "seed": cfg.seed,
        "dry_run": cfg.dry_run,
        "n_cleaning_steps": state.cleaning_steps,
        "n_fe_steps": state.fe_steps,
        "final_schema": final_schema,
        "lib_versions": _lib_versions(),
    }

    if cfg.dry_run or state.splits is None:
        metadata["note"] = "dry-run : recette produite, partitions non matérialisées."
        _write_json(output_dir / "metadata.json", metadata)
        return output_dir

    # Partitions + dtypes (le CSV ne stocke pas les types ; dtypes.json les réimpose).
    splits = state.splits
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    _write_json(output_dir / "dtypes.json", _dtypes_map(df))

    metadata["split_ratios"] = {
        "mode": DEFAULT_MODE,
        "test_size": DEFAULT_TEST_SIZE,
        **({"val_size": DEFAULT_VAL_SIZE} if DEFAULT_MODE == "3way" else {}),
    }
    metadata["splits"] = {
        name: {"file": f"{name}.csv", "n_rows": int(len(frame))}
        for name, frame in splits.items()
    }
    _write_json(output_dir / "metadata.json", metadata)
    return output_dir
