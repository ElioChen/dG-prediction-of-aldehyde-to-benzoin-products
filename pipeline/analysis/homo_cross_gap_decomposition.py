#!/usr/bin/env python
"""Task B: is the homo (1.503) vs cross (2.215) dG MAE gap 'resolved'?

The two headline numbers are not the same ruler:
  * homo 1.503  -- random 70/10/20 split, closed 220k library (interpolation)
  * cross 2.215 -- Bemis-Murcko scaffold-disjoint holdout (extrapolation), blend,
                   clean-train 22,771

This script does the matched comparison the project never ran, on the
git-tracked `homo_unify_v1` subset (30,000 homo pairs that DO carry
dG_orca_kcal + a scaffold-disjoint split + the g-xTB baseline):

  homo_random    : random 70/10/20 (seed 42) of the 30k          -> homo, interpolation regime
  homo_scaffold  : the provided scaffold-disjoint split           -> homo, extrapolation regime
                   (train ~21,967 ~= cross clean-train 22,771 -- already size-matched)

Same 72-feature Delta-learning recipe and same 3-member ensemble
(MLP + XGB-d8 + XGB-d10) as pipeline/analysis/finalize_correction.py, so the
only things that move between homo_random and homo_scaffold are the split
regime; between homo_scaffold and cross are the task (3-species dG vs ...
still 3-species, but cross donor != acceptor) and the label-noise floor.

Decomposition reported:
  leakage_premium_homo   = homo_scaffold_MAE - homo_random_MAE
  regime_matched_gap     = homo_scaffold_MAE - cross_ensemble_only_MAE(2.326)
  + label-noise floors (homo ~2.1, cross ~2.9) and delta/label spreads for context

Runs on CPU (sklearn + xgboost + rdkit), ~10-15 min. No SLURM needed.
  /home/schen3/venv/nhc-workflow/bin/python pipeline/analysis/homo_cross_gap_decomposition.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]
HU = REPO / "data/cross_benzoin/homo_unify"
HV6 = REPO / "data/cross_benzoin/homo_v6"
OUT = REPO / "data/analysis/homo_cross_gap"
OUT.mkdir(parents=True, exist_ok=True)

PROD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega",
  "xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH",
  "wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC",
  "fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L",
  "sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp = [f"ald_{c}" for c in ALD]
GKEYS = ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing",
         "nAmide","has_P","has_B","has_S","has_Si","has_halogen"]
GLOB = [f"g_{k}" for k in GKEYS]
FEATS = PROD_QM + ALDp + GLOB  # 72

# reference cross r1-10 scaffold-disjoint numbers (CHAMPION.md / cross_r1_10 memory)
CROSS = {"blend_mae": 2.215, "ensemble_only_mae": 2.326, "gnn_only_mae": 2.313,
         "single_xgb_mae": 2.548, "gxtb_baseline_mae": 5.037, "clean_train": 22771,
         "holdout_n": 448, "conformer_noise_floor_kcal": 2.88}
HOMO_HEADLINE = {"random_split_mae_full_219k": 1.503, "conformer_noise_floor_kcal_bulk": 0.92,
                 "conformer_noise_floor_kcal_k5_34mol": 2.14}


def gfeats(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return {f"g_{k}": np.nan for k in GKEYS}
    s = {a.GetSymbol() for a in m.GetAtoms()}
    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m), int("P" in s), int("B" in s), int("S" in s),
            int("Si" in s), int(bool(s & {"F", "Cl", "Br", "I"}))]
    return {f"g_{k}": v for k, v in zip(GKEYS, vals)}


def add_global(df, smi_col="smiles"):
    u = df[[smi_col]].drop_duplicates()
    g = pd.DataFrame([gfeats(s) for s in u[smi_col]])
    g[smi_col] = u[smi_col].values
    return df.merge(g, on=smi_col, how="left")


def _xgb(depth, ne):
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                        early_stopping_rounds=60, eval_metric="mae")


def fit_eval(df, tr, va, te, tag):
    sc = StandardScaler().fit(df[FEATS].values[tr])
    Xtr, Xva, Xte = (sc.transform(df[FEATS].values[i]) for i in (tr, va, te))
    dtr, dva = df["delta"].values[tr], df["delta"].values[va]
    gte, yte = df["dG_gxtb_kcal"].values[te], df["dG_orca_kcal"].values[te]
    members = [("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                                    early_stopping=True, n_iter_no_change=12)),
               ("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]
    preds = []
    for name, mdl in members:
        if name == "MLP":
            mdl.fit(Xtr, dtr)
        else:
            mdl.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
        preds.append(mdl.predict(Xte))
    dpred = np.mean(preds, axis=0)
    ypred = gte + dpred
    mae = float(mean_absolute_error(yte, ypred))
    rmse = float(np.sqrt(np.mean((yte - ypred) ** 2)))
    r2 = float(r2_score(yte, ypred))
    base_mae = float(mean_absolute_error(yte, gte))
    print(f"  [{tag}] n_tr={len(tr)} n_te={len(te)}  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} "
          f"| g-xTB baseline MAE={base_mae:.3f}", flush=True)
    return {"tag": tag, "n_train": len(tr), "n_test": len(te), "mae": mae, "rmse": rmse,
            "r2": r2, "gxtb_baseline_mae": base_mae}


def main() -> int:
    t0 = time.time()
    dft = pd.read_csv(HU / "homo_unify_v1_dft.csv")
    spl = pd.read_csv(HU / "homo_unify_v1_scaffold_split_lookup.csv")
    prod = pd.read_csv(HU / "homo_unify_v1_products.csv", low_memory=False)
    ald = (pd.read_csv(HV6 / "aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False)
             .drop_duplicates("id")
             .rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}}))

    df = prod.merge(dft, on="id").merge(spl[["id", "scaffold_split"]], on="id")
    df["ald_id"] = df["donor_id"].astype("Int64")
    df = df.merge(ald, on="ald_id", how="left")
    df = add_global(df, "smiles")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows with all 72 feats: {len(df):,}  "
          f"split: {df['scaffold_split'].value_counts().to_dict()}", flush=True)

    results = {}

    # (1) scaffold-disjoint split as provided
    tr = df.index[df.scaffold_split == "train"].to_numpy()
    va = df.index[df.scaffold_split == "validation"].to_numpy()
    te = df.index[df.scaffold_split == "test"].to_numpy()
    results["homo_scaffold"] = fit_eval(df, tr, va, te, "homo_scaffold")

    # (2) random 70/10/20 of the same rows, seed 42 (finalize_correction.py convention)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df))
    results["homo_random"] = fit_eval(df, idx[:ntr], idx[ntr:nva], idx[nva:], "homo_random")

    # (3) scaffold-disjoint but train subsampled to cross clean-train size (already
    # ~matched: 21,967 vs 22,771 -- run it anyway so the number is explicit)
    k = min(CROSS["clean_train"], len(tr))
    tr_sub = rng.choice(tr, size=k, replace=False)
    results["homo_scaffold_sizematched"] = fit_eval(df, tr_sub, va, te, "homo_scaffold_sizematched")

    leak = results["homo_scaffold"]["mae"] - results["homo_random"]["mae"]
    regime_gap = results["homo_scaffold"]["mae"] - CROSS["ensemble_only_mae"]
    out = {
        "built_by": "pipeline/analysis/homo_cross_gap_decomposition.py",
        "dataset": "homo_unify_v1 (30k git-tracked homo subset with dG_orca + scaffold-disjoint split)",
        "recipe": "72-feature Delta-learning, MLP + XGB-d8 + XGB-d10 mean ensemble (== finalize_correction.py)",
        "label_spread": {"delta_std_kcal": float(df["delta"].std()),
                          "dG_orca_std_kcal": float(df["dG_orca_kcal"].std())},
        "results": results,
        "reference": {"homo_headline": HOMO_HEADLINE, "cross_r1_10": CROSS},
        "decomposition": {
            "leakage_premium_homo_kcal": leak,
            "leakage_premium_homo_pct": 100 * leak / results["homo_random"]["mae"],
            "regime_matched_gap_vs_cross_ensemble_only_kcal": regime_gap,
            "note": ("homo_scaffold vs homo_random isolates the interpolation->extrapolation "
                     "cost at ~22k train on the homo task; homo_scaffold vs cross ensemble-only "
                     "(2.326, the no-GNN comparator) is what remains after matching split "
                     "regime and train size -- attributable to task difficulty + the higher "
                     "cross label-noise floor (~2.9 vs homo ~2.1 kcal single-conformer)."),
        },
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "homo_cross_gap_decomposition.json").write_text(json.dumps(out, indent=2))
    joblib.dump(df[["id", "scaffold_split", "delta", "dG_orca_kcal", "dG_gxtb_kcal"]],
                OUT / "homo_unify_labeled_frame.joblib")
    print("\n" + json.dumps(out["decomposition"], indent=2))
    print(f"\n-> {OUT / 'homo_cross_gap_decomposition.json'}  ({out['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
