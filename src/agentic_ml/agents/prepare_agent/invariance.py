"""Test d'invariance par ligne — le gate obligatoire (§6).

C'est ce test déterministe (et non le jugement du LLM) qui garantit qu'une
transformation est *row-local* / stateless. Une transformation `f` est row-local
ssi la sortie d'une ligne ne dépend **pas** des autres lignes présentes. On le
vérifie en comparant `f` appliquée à plusieurs contextes (l'échantillon complet,
des sous-échantillons, éventuellement une permutation) : pour chaque ligne
identifiable, la sortie doit être strictement identique.

### Stratégie de clé d'alignement (§6.3)
Les lignes sont identifiées par l'**index** pandas de l'échantillon (figé en
RangeIndex au tirage). L'index est transparent pour les opérations basées sur les
valeurs (`drop_duplicates`, filtres, calculs colonne) : il n'altère donc pas leur
sémantique, contrairement à une colonne-clé injectée. Une transformation qui
réinitialise l'index (`reset_index(drop=True)`) brise l'alignement → le gate
échoue avec un diagnostic explicite, ce qui est le comportement voulu (fail-safe).

### Survie des lignes
Le gate vérifie aussi que la **décision de garder/supprimer** une ligne est
row-local : une ligne présente dans le contexte de référence et dans le
sous-échantillon doit survivre dans les deux ou dans aucun. Cela attrape les
filtres globaux (« top 10 % », outliers IQR/z-score, etc.).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agentic_ml.config import (
    GATE_FLOAT_ATOL,
    GATE_FLOAT_RTOL,
    GATE_PERMUTATION_SEED,
    GATE_SAMPLE_SEED,
    GATE_SAMPLE_SIZE,
    GATE_SUBSAMPLES,
)

from agentic_ml.agents.prepare_agent.executor import ExecutorError, TransformFn
from agentic_ml.agents.prepare_agent.state import InvarianceCheck, InvarianceReport


def _bounded_sample(df: pd.DataFrame, *, size: int, seed: int) -> pd.DataFrame:
    """Tire un échantillon borné `E` et fige une clé stable (RangeIndex 0..m-1)."""
    if len(df) > size:
        e = df.sample(n=size, random_state=seed)
    else:
        e = df
    return e.reset_index(drop=True)


def _scalar_equal(a, b, *, atol: float, rtol: float) -> bool:
    """Égalité tolérante aux NaN ; tolérance flottante seulement si atol/rtol > 0."""
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na or b_na:
        return bool(a_na and b_na)
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(
        b, (int, float, np.floating, np.integer)
    ):
        if atol == 0.0 and rtol == 0.0:
            return a == b
        return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return a == b


def _compare(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    input_keys: list,
    *,
    atol: float,
    rtol: float,
) -> Optional[str]:
    """Compare `other` à `ref` sur les lignes identifiées par `input_keys`.

    `input_keys` sont les clés présentes en **entrée** du contexte `other` (toutes
    les clys de E pour la référence/permutation ; les clés du sous-échantillon
    sinon). Renvoie `None` si tout est identique, sinon un diagnostic.
    """
    if ref.index.has_duplicates or other.index.has_duplicates:
        return "La transformation a dupliqué des lignes (index non unique) : non row-local."

    ref_surv = set(ref.index)
    other_surv = set(other.index)

    extra = other_surv - set(input_keys)
    if extra:
        return (
            "La transformation a produit des lignes absentes de l'entrée "
            f"(clé exemple={next(iter(extra))}) : l'index a probablement été réinitialisé."
        )

    # Cohérence de survie : garder/supprimer doit être une décision row-local.
    for key in input_keys:
        if (key in ref_surv) != (key in other_surv):
            kept_ref = key in ref_surv
            return (
                f"La ligne clé={key} survit dans un contexte mais pas dans l'autre "
                f"(référence={'gardée' if kept_ref else 'supprimée'}, "
                f"autre={'gardée' if not kept_ref else 'supprimée'}) : "
                "la décision de filtrage dépend du contexte (non row-local)."
            )

    common = [k for k in input_keys if k in ref_surv and k in other_surv]
    if not common:
        return None  # rien à comparer (ex. toutes les lignes filtrées identiquement)

    shared_cols = [c for c in ref.columns if c in other.columns]
    for col in shared_cols:
        ref_col = ref.loc[common, col]
        oth_col = other.loc[common, col]
        for key in common:
            if not _scalar_equal(ref_col.at[key], oth_col.at[key], atol=atol, rtol=rtol):
                return (
                    f"Sortie divergente colonne « {col} », ligne clé={key} : "
                    f"référence={ref_col.at[key]!r} vs autre={oth_col.at[key]!r}. "
                    "Une statistique agrégée du dataset est probablement utilisée."
                )
    return None


def run_invariance_gate(
    fn: TransformFn,
    df: pd.DataFrame,
    *,
    enable_permutation: bool = False,
    sample_size: int = GATE_SAMPLE_SIZE,
    subsamples: tuple = GATE_SUBSAMPLES,
    atol: float = GATE_FLOAT_ATOL,
    rtol: float = GATE_FLOAT_RTOL,
) -> InvarianceReport:
    """Soumet `fn` au test d'invariance par ligne. Aucune mutation de `df`.

    Le coût propre du gate est constant : seul le calcul de la référence est en
    `O(échantillon)`, jamais en `O(N)`.
    """
    e = _bounded_sample(df, size=sample_size, seed=GATE_SAMPLE_SEED)
    all_keys = list(e.index)
    checks: list[InvarianceCheck] = []

    def _run(frame: pd.DataFrame) -> pd.DataFrame:
        out = fn(frame.copy())
        if not isinstance(out, pd.DataFrame):
            raise ExecutorError("`transform` doit renvoyer un DataFrame.")
        return out

    # 1. Déterminisme : deux exécutions identiques sur E.
    ref = _run(e)
    ref2 = _run(e)
    diag = _compare(ref, ref2, all_keys, atol=atol, rtol=rtol)
    checks.append(InvarianceCheck(kind="determinism", passed=diag is None))
    if diag is not None:
        return InvarianceReport(
            passed=False, sample_size=len(e), checks=checks,
            diagnostic=f"[déterminisme] {diag}",
        )

    # 2. Sous-échantillons (noyau obligatoire §6.2) : attrape les statistiques agrégées.
    for frac, seed in subsamples:
        s = e.sample(frac=frac, random_state=seed)
        rs = _run(s)
        diag = _compare(ref, rs, list(s.index), atol=atol, rtol=rtol)
        checks.append(
            InvarianceCheck(kind="subsample", fraction=frac, seed=seed, passed=diag is None)
        )
        if diag is not None:
            return InvarianceReport(
                passed=False, sample_size=len(e), checks=checks,
                diagnostic=f"[sous-échantillon frac={frac} seed={seed}] {diag}",
            )

    # 3. Permutation (optionnel §6.2) : attrape les dépendances d'ordre/voisinage.
    if enable_permutation:
        e_perm = e.sample(frac=1.0, random_state=GATE_PERMUTATION_SEED)
        rp = _run(e_perm)
        diag = _compare(ref, rp, all_keys, atol=atol, rtol=rtol)
        checks.append(
            InvarianceCheck(kind="permutation", seed=GATE_PERMUTATION_SEED, passed=diag is None)
        )
        if diag is not None:
            return InvarianceReport(
                passed=False, sample_size=len(e), checks=checks,
                diagnostic=f"[permutation] {diag}",
            )

    return InvarianceReport(passed=True, sample_size=len(e), checks=checks, diagnostic=None)
