"""Persistance d'une run de preprocessing dans `data/01_preproc/<dataset>_NNN/`.

Écrit :
- `preprocessed.csv`    : le DataFrame final transformé ;
- `transformations.json`: l'historique complet (transformations + features) ;
- `analysis_report.json`: le dernier rapport de l'Agent Analyse.

La numérotation incrémentale (`iris_001`, `iris_002`, …) reprend la convention
de `data_manager.DataSplitter`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agentic_ml.agents.preproc_agent.state import MLPipelineState
from agentic_ml.config import PREPROC_DATA_DIR, RUN_FOLDER_WIDTH

logger = logging.getLogger("agentic_ml.agents.preproc_agent")


def _next_run_folder(output_dir: Path, dataset_name: str) -> Path:
    """Renvoie le prochain dossier `<dataset>_NNN` disponible."""
    prefix = f"{dataset_name}_"
    existing = [
        int(p.name[len(prefix):])
        for p in output_dir.iterdir()
        if p.is_dir()
        and p.name.startswith(prefix)
        and p.name[len(prefix):].isdigit()
    ]
    next_idx = max(existing, default=0) + 1
    return output_dir / f"{prefix}{str(next_idx).zfill(RUN_FOLDER_WIDTH)}"


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def persist_results(
    state: MLPipelineState,
    *,
    output_dir: Path = PREPROC_DATA_DIR,
    dataset_name: str = "iris",
) -> Path:
    """Écrit les artefacts de la run et renvoie le dossier créé."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_folder = _next_run_folder(output_dir, dataset_name)
    run_folder.mkdir()

    df = state["dataframe"]
    df.to_csv(run_folder / "preprocessed.csv", index=False)

    _write_json(
        run_folder / "transformations.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "agent_model": state.get("agent_model"),
            "target_column": state.get("target_column"),
            "iteration_count": state.get("iteration_count", 0),
            "applied_transformations": state.get("applied_transformations", []),
            "created_features": state.get("created_features", []),
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        },
    )
    _write_json(run_folder / "analysis_report.json", state.get("analysis_report", {}))

    logger.info("persist | données preprocessées écrites dans %s", run_folder)
    return run_folder
