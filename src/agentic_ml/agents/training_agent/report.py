"""Mise en forme lisible du rapport de classification par classe.

`val_class_report` est le dict renvoyé par `classification_report(output_dict=True)`
de sklearn : une entrée par classe (precision/recall/f1-score/support), plus les
agrégats `accuracy`, `macro avg` et `weighted avg`. Fonctionne pour 2, 3 ou N classes.
"""
from __future__ import annotations

from typing import Any

# Agrégats à exclure de la liste des classes (traités séparément).
_AGGREGATES = ("accuracy", "macro avg", "weighted avg")


def format_class_report(report: dict[str, Any], *, indent: str = "  ") -> str:
    """Rend le rapport par classe sous forme de tableau aligné multi-lignes."""
    if not report:
        return f"{indent}(aucun rapport par classe disponible)"

    classes = [k for k in report if k not in _AGGREGATES]
    label_width = max((len(str(c)) for c in classes + list(_AGGREGATES)), default=7)

    header = (
        f"{indent}{'classe':<{label_width}}  "
        f"{'precision':>9}  {'recall':>7}  {'f1-score':>8}  {'support':>7}"
    )
    lines = [header]

    def _row(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"{indent}{label:<{label_width}}  "
            f"{metrics['precision']:>9.4f}  {metrics['recall']:>7.4f}  "
            f"{metrics['f1-score']:>8.4f}  {int(metrics['support']):>7d}"
        )

    for cls in classes:
        lines.append(_row(str(cls), report[cls]))

    if "accuracy" in report:
        lines.append(f"{indent}{'accuracy':<{label_width}}  {report['accuracy']:>9.4f}")
    for agg in ("macro avg", "weighted avg"):
        if agg in report:
            lines.append(_row(agg, report[agg]))

    return "\n".join(lines)
