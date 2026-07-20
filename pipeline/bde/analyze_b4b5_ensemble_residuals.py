#!/usr/bin/env python
"""Phase-1 next-step #3, first two sub-items (2026-07-15): does a B4+B5 blend beat either
model alone, and where does the remaining error concentrate?

Needs pipeline/bde/train_gnn_bde.py (B4) and train_bondnet_bde.py (B5) to have been rerun
with --pred-out (the original runs only saved summary JSON, no per-row predictions -- see
PROGRESS_20260714.md "下一次会话请从这里开始" item 3). B4 and B5 use the same
molecule_cold_split(seed=0, test_frac=0.15) on the same split column per `which`, so their
test sets are (near-)identical; this script inner-joins on id to be safe against the odd
row that QC-dropped in only one run.

Blend weight is chosen on a held-out half of the test set and reported on the other half
(not fit-and-report on the same rows) so the reported MAE isn't an optimistic re-fit --
mirrors how the main dG model's GNN+tabular stacking weight was meant to be validated
(BDE_prediction.md "5. 架构提升尝试": -0.051 MAE real gain, small but real).

Functional-group tags reuse the EWG/heteroatom SMARTS set from
pipeline/analysis/screen_v6_funcgroup_analysis.py (same groups the main dG model found
enriched in its worst residuals: sulfonyl/phosphorus/imine-adjacent), extended with a
couple of common organic groups relevant to aldehyde/product BDE (ester, amide, ether,
halogen) to broaden coverage beyond just the xTB-unreliable set.

Usage:
  python analyze_b4b5_ensemble_residuals.py --out /tmp/b4b5_ensemble.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, r2_score

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
LOGS = Path("/scratch-shared/schen3/benzoin-dg/runs/logs")

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


def blend_eval(y, p4, p5, seed=0):
    """Grid-search blend weight on a random half of rows, report MAE on the other half."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    sel, rep = idx[:half], idx[half:]
    alphas = np.linspace(0, 1, 21)
    mae_sel = [mean_absolute_error(y[sel], a * p4[sel] + (1 - a) * p5[sel]) for a in alphas]
    best_alpha = float(alphas[int(np.argmin(mae_sel))])
    pred_rep = best_alpha * p4[rep] + (1 - best_alpha) * p5[rep]
    return {
        "best_alpha_B4weight": best_alpha,
        "blend_MAE_heldout_half": float(mean_absolute_error(y[rep], pred_rep)),
        "blend_R2_heldout_half": float(r2_score(y[rep], pred_rep)),
        "naive_50_50_MAE_heldout_half": float(
            mean_absolute_error(y[rep], 0.5 * p4[rep] + 0.5 * p5[rep])),
        "B4_MAE_heldout_half": float(mean_absolute_error(y[rep], p4[rep])),
        "B5_MAE_heldout_half": float(mean_absolute_error(y[rep], p5[rep])),
    }


def analyze_one(which: str):
    p4 = pd.read_csv(LOGS / f"bde_gnn_{which}_test_predictions.csv", dtype={"id": str})
    p5 = pd.read_csv(LOGS / f"bondnet_bde_{which}_test_predictions.csv", dtype={"id": str})
    m = p4.merge(p5, on="id", suffixes=("_B4", "_B5"))
    assert np.allclose(m["y_true_B4"], m["y_true_B5"], atol=1e-6), \
        "B4/B5 disagree on label for the same id -- splits are not actually aligned"
    y = m["y_true_B4"].to_numpy()
    p4a, p5a = m["y_pred_B4"].to_numpy(), m["y_pred_B5"].to_numpy()
    print(f"{which}: {len(m)} ids in both B4 and B5 test sets", flush=True)

    result = {"which": which, "n_shared_test": len(m),
              "B4_MAE_full": float(mean_absolute_error(y, p4a)),
              "B5_MAE_full": float(mean_absolute_error(y, p5a))}
    result["blend"] = blend_eval(y, p4a, p5a)

    id_col = "id" if which == "aldehydes" else "donor_id"
    mol_cols = ["id", "smiles"] if which == "aldehydes" else ["id", "donor_id", "smiles"]
    mol = pd.read_csv(H / f"{which}_all.csv", usecols=mol_cols, dtype=str,
                       keep_default_na=False)
    m = m.merge(mol, on="id", how="left")
    tags = tag_groups(m["smiles"])
    m = pd.concat([m, tags], axis=1)

    alpha = result["blend"]["best_alpha_B4weight"]
    m["ensemble_pred"] = alpha * m["y_pred_B4"] + (1 - alpha) * m["y_pred_B5"]
    m["abs_err"] = (m["ensemble_pred"] - m["y_true_B4"]).abs()
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
            "enrichment_vs_overall_mean": round(float(m.loc[mask, "abs_err"].mean()) /
                                                 overall_mae, 3),
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
        p4_path = LOGS / f"bde_gnn_{which}_test_predictions.csv"
        p5_path = LOGS / f"bondnet_bde_{which}_test_predictions.csv"
        if not (p4_path.exists() and p5_path.exists()):
            print(f"skip {which}: missing {p4_path if not p4_path.exists() else p5_path} "
                  f"(rerun train_gnn_bde.py / train_bondnet_bde.py with --pred-out first)")
            continue
        out[which] = analyze_one(which)

    print(json.dumps(out, indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
