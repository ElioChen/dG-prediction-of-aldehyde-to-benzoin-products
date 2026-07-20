#!/usr/bin/env python
"""Generalizes analyze_b4b5_ensemble_residuals.py to a 3-way B4+B5+B6 blend, now that
B6 (train_gnn_hybrid_bde.py, GNN+3D-descriptor fusion) is in and dominates both B4 and B5
standalone (aldehydes MAE 1.60/1.88 -> 1.10, products 3.49-3.64 -> 2.12 as of 2026-07-15).
Grid-searches the 3-way simplex weight on a held-out half of the shared test set, reports
on the other half -- same held-out-half protocol as the original 2-way script, so the
reported MAE isn't an optimistic re-fit.

Usage:
  python analyze_b456_ensemble_residuals.py --out /tmp/b456_ensemble.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, r2_score

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
LOGS = Path("/scratch-shared/schen3/benzoin-dg/runs/logs")

MODELS = {"B4": "bde_gnn", "B5": "bondnet_bde", "B6": "bde_gnn_hybrid"}

GROUPS = [
    ("sulfonyl", "[#16X4](=[OX1])(=[OX1])"),
    ("sulfonyl_fluoride", "[#16X4](=[OX1])(=[OX1])[F]"),
    ("triflate", "[OX2][#16X4](=[OX1])(=[OX1])[CX4]([F])([F])[F]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("n_oxide", "[#7+][#8X1-]"),
    ("imine", "[CX3]=[NX2]"),
    ("boron", "[#5]"),
    ("phosphorus", "[#15]"),
    ("selenium", "[#34]"),
    ("silicon", "[#14]"),
    ("ester", "[#6][CX3](=O)[OX2][#6]"),
    ("amide", "[NX3][CX3](=[OX1])"),
    ("ether", "[OD2]([#6])[#6]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("tert_alkyl", "[CX4]([#6])([#6])([#6])[#6]"),
]
PATS = [(lab, Chem.MolFromSmarts(sm)) for lab, sm in GROUPS]


def tag_groups(smiles: pd.Series) -> pd.DataFrame:
    flags = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        flags.append([bool(m and m.HasSubstructMatch(p)) for _, p in PATS])
    return pd.DataFrame(flags, columns=[g[0] for g in GROUPS], index=smiles.index)


def simplex_grid(n_models: int, step: float = 0.1):
    """All weight tuples in [0,1]^n_models summing to 1, on a `step` grid."""
    n_steps = int(round(1.0 / step))
    for combo in itertools.product(range(n_steps + 1), repeat=n_models - 1):
        if sum(combo) > n_steps:
            continue
        last = n_steps - sum(combo)
        yield tuple(c * step for c in combo) + (last * step,)


def blend_eval(y, preds: dict, seed=0):
    names = list(preds.keys())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    sel, rep = idx[:half], idx[half:]

    best_w, best_mae = None, np.inf
    for w in simplex_grid(len(names), step=0.1):
        pred_sel = sum(wi * preds[n][sel] for wi, n in zip(w, names))
        mae = mean_absolute_error(y[sel], pred_sel)
        if mae < best_mae:
            best_mae, best_w = mae, w

    pred_rep = sum(wi * preds[n][rep] for wi, n in zip(best_w, names))
    out = {
        "weights": dict(zip(names, [round(wi, 2) for wi in best_w])),
        "blend_MAE_heldout_half": float(mean_absolute_error(y[rep], pred_rep)),
        "blend_R2_heldout_half": float(r2_score(y[rep], pred_rep)),
        "naive_equal_MAE_heldout_half": float(mean_absolute_error(
            y[rep], sum(preds[n][rep] for n in names) / len(names))),
    }
    for n in names:
        out[f"{n}_MAE_heldout_half"] = float(mean_absolute_error(y[rep], preds[n][rep]))
    return out


def analyze_one(which: str):
    dfs = {}
    for tag, fname in MODELS.items():
        p = LOGS / f"{fname}_{which}_test_predictions.csv"
        if not p.exists():
            print(f"  {which}: missing {tag} ({p}) -- skipping this split", flush=True)
            return None
        dfs[tag] = pd.read_csv(p, dtype={"id": str})

    m = dfs["B4"][["id", "y_true"]].copy()
    for tag, df in dfs.items():
        m = m.merge(df[["id", "y_pred"]].rename(columns={"y_pred": f"y_pred_{tag}"}), on="id")
        assert np.allclose(m["y_true"], df.set_index("id").loc[m["id"], "y_true"], atol=1e-6), \
            f"{tag} disagrees on label for a shared id -- splits are not actually aligned"
    y = m["y_true"].to_numpy()
    preds = {tag: m[f"y_pred_{tag}"].to_numpy() for tag in MODELS}
    print(f"{which}: {len(m)} ids shared across B4/B5/B6 test sets", flush=True)

    result = {"which": which, "n_shared_test": len(m)}
    for tag in MODELS:
        result[f"{tag}_MAE_full"] = float(mean_absolute_error(y, preds[tag]))
    result["blend"] = blend_eval(y, preds)

    id_col = "id" if which == "aldehydes" else "donor_id"
    mol_cols = ["id", "smiles"] if which == "aldehydes" else ["id", "donor_id", "smiles"]
    mol = pd.read_csv(H / f"{which}_all.csv", usecols=mol_cols, dtype=str, keep_default_na=False)
    m = m.merge(mol, on="id", how="left")
    tags = tag_groups(m["smiles"])
    m = pd.concat([m, tags], axis=1)

    w = result["blend"]["weights"]
    m["ensemble_pred"] = sum(w[tag] * m[f"y_pred_{tag}"] for tag in MODELS)
    m["abs_err"] = (m["ensemble_pred"] - m["y_true"]).abs()
    overall_mae = float(m["abs_err"].mean())

    group_stats = {}
    for lab, _ in PATS:
        mask = m[lab]
        n = int(mask.sum())
        if n < 20:
            continue
        group_stats[lab] = {
            "n": n, "share_pct": round(100 * n / len(m), 2),
            "median_abs_err": float(m.loc[mask, "abs_err"].median()),
            "mean_abs_err": float(m.loc[mask, "abs_err"].mean()),
            "enrichment_vs_overall_mean": round(float(m.loc[mask, "abs_err"].mean()) / overall_mae, 3),
        }
    result["overall_ensemble_MAE"] = overall_mae
    result["functional_group_residuals"] = dict(
        sorted(group_stats.items(), key=lambda kv: -kv[1]["enrichment_vs_overall_mean"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    for which in ["aldehydes", "products"]:
        r = analyze_one(which)
        if r is not None:
            out[which] = r

    print(json.dumps(out, indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
