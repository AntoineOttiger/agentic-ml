"""Recette rejouable (§7) : sérialisation et ré-application sans l'agent.

L'historique n'est pas un log en prose mais une recette exécutable : un
`transformations.json` (métadonnées + référence) accompagné d'un dossier
`transforms/` contenant chaque script `.py` versionné. `replay_recipe` reconstruit
la pipeline cleaning+FE sur un nouveau CSV de même schéma — sans aucun appel LLM.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from agentic_ml.agents.prepare_agent.executor import apply_transform, compile_transform
from agentic_ml.agents.prepare_agent.state import TransformationRecord

_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")


def script_path_for(index: int, phase: str, name: str) -> str:
    """Chemin relatif déterministe du script versionné d'une étape."""
    slug = _SLUG_RE.sub("_", name).strip("_").lower() or "step"
    return f"transforms/{index:02d}_{phase}_{slug}.py"


def write_recipe(history: list[TransformationRecord], output_dir: Path) -> Path:
    """Écrit `transforms/*.py` + `transformations.json`. Renvoie le json."""
    transforms_dir = output_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    for record in history:
        (output_dir / record.script_path).write_text(record.code, encoding="utf-8")

    payload = [record.model_dump() for record in history]
    json_path = output_dir / "transformations.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return json_path


def replay_recipe(recipe_dir: Path | str, df: pd.DataFrame) -> pd.DataFrame:
    """Rejoue la recette d'un répertoire de sortie sur un dataframe `df`.

    Les étapes sont appliquées dans l'ordre, à partir du code source versionné —
    garantissant la reproductibilité de l'acceptation (§12) sans l'agent.
    """
    recipe_dir = Path(recipe_dir)
    records = json.loads(
        (recipe_dir / "transformations.json").read_text(encoding="utf-8")
    )
    out = df
    for record in records:
        code = (recipe_dir / record["script_path"]).read_text(encoding="utf-8")
        out = apply_transform(compile_transform(code), out)
    return out
