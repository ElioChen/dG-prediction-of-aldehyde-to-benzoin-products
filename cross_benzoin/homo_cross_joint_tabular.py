#!/usr/bin/env python
"""Task C (cheap tabular first pass): does adding homo dG Delta examples help the
cross tabular model at CURRENT scale?

"Open problem 1" (homo+cross joint training for the dG task) was last tested at
<=17k cross rows and shelved as a shrinking benefit. Cross now has ~23k clean-
train and 30k homo rows carry real DFT labels (homo_unify_v1). This re-runs the
comparison at that scale, on the 72-feature space the two tasks share:

  34 product QM  (same column names in both tables)
  22 aldehyde QM (homo ald_X  <->  cross donor_X ; homo donor==acceptor)
  16 RDKit global on the product SMILES (recomputed for both)

target = delta = dG_orca_kcal - dG_gxtb_kcal ; eval on the cross scaffold-
disjoint holdout (n=448). Three XGB conditions:

  cross_only    XGB on cross clean-train only            (this experiment's baseline)
  naive_merge   XGB on homo_unify + cross, + is_homo flag
  finetune      XGB on homo_unify, then continue-train (xgb_model=) on cross

GREEN if naive_merge or finetune beats cross_only by > ~0.10 kcal (then a full
260-schema homo featurization + GNN homo-pretrain is worth it); RED if not (the
lever stays closed, matching the pre-purge finding at smaller scale).

  /home/schen3/venv/nhc-workflow/bin/python cross_benzoin/homo_cross_joint_tabular.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[1]
HU = REPO / "data/cross_benzoin/homo_unify"
HV6 = REPO / "data/cross_benzoin/homo_v6"
XTABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
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
FEATS = PROD_QM + ALDp + GLOB
CROSS_REF = {"ensemble_only_mae": 2.326, "blend_mae": 2.215, "single_xgb_mae": 2.548,
             "gxtb_baseline_mae": 5.037}


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


def add_global(df):
    u = df[["smiles"]].drop_duplicates()
    g = pd.DataFrame([gfeats(s) for s in u["smiles"]])
    g["smiles"] = u["smiles"].values
    return df.merge(g, on="smiles", how="left")


def _xgb():
    return XGBRegressor(n_estimators=1800, max_depth=9, learning_rate=0.02, subsample=0.75,
                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16, eval_metric="mae")


def load_homo() -> pd.DataFrame:
    dft = pd.read_csv(HU / "homo_unify_v1_dft.csv")
    prod = pd.read_csv(HU / "homo_unify_v1_products.csv", low_memory=False)
    ald = (pd.read_csv(HV6 / "aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False)
             .drop_duplicates("id")
             .rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}}))
    df = prod.merge(dft, on="id")
    df["ald_id"] = df["donor_id"].astype("Int64")
    df = df.merge(ald, on="ald_id", how="left")
    df = add_global(df)
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    df["is_homo"] = 1
    return df


def load_cross() -> pd.DataFrame:
    df = pd.read_parquet(XTABLE)
    ren = {f"donor_{c}": f"ald_{c}" for c in ALD}
    df = df.rename(columns=ren)
    df = add_global(df)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    df["is_homo"] = 0
    keep = ["id", "new_scaffold_split", "delta", "dG_orca_kcal", "dG_gxtb_kcal", "is_homo"] + FEATS
    return df[keep]


def _ev(model, te):
    dp = model.predict(te[FEATS + ["is_homo"]].to_numpy())
    yp = te["dG_gxtb_kcal"].to_numpy() + dp
    yt = te["dG_orca_kcal"].to_numpy()
    return {"mae": float(mean_absolute_error(yt, yp)), "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def main() -> int:
    t0 = time.time()
    homo = load_homo()
    cross = load_cross()
    xtr = cross[cross.new_scaffold_split == "train"].reset_index(drop=True)
    xte = cross[cross.new_scaffold_split == "test"].reset_index(drop=True)
    print(f"homo {len(homo)}  cross clean-train {len(xtr)}  cross holdout {len(xte)}", flush=True)
    F = FEATS + ["is_homo"]

    res = {}

    m = _xgb(); m.fit(xtr[F].to_numpy(), xtr["delta"].to_numpy())
    res["cross_only"] = _ev(m, xte)
    print("cross_only", res["cross_only"], flush=True)

    both = pd.concat([homo[F + ["delta"]], xtr[F + ["delta"]]], ignore_index=True)
    m = _xgb(); m.fit(both[F].to_numpy(), both["delta"].to_numpy())
    res["naive_merge"] = _ev(m, xte)
    print("naive_merge", res["naive_merge"], flush=True)

    m = _xgb(); m.fit(homo[F].to_numpy(), homo["delta"].to_numpy())
    res["homo_only_zeroshot"] = _ev(m, xte)
    m2 = _xgb(); m2.fit(xtr[F].to_numpy(), xtr["delta"].to_numpy(), xgb_model=m.get_booster())
    res["finetune"] = _ev(m2, xte)
    print("homo_only_zeroshot", res["homo_only_zeroshot"], flush=True)
    print("finetune", res["finetune"], flush=True)

    base = res["cross_only"]["mae"]
    best_transfer = min(res["naive_merge"]["mae"], res["finetune"]["mae"])
    gain = base - best_transfer
    if gain > 0.10:
        verdict = (f"GREEN: homo transfer helps ({gain:+.3f} kcal over cross_only 72-feat XGB "
                   f"{base:.3f}). Worth building the full 260-schema homo featurization + "
                   f"GNN homo-pretrain->finetune.")
    elif gain > 0.03:
        verdict = (f"AMBER: marginal homo-transfer gain ({gain:+.3f} kcal). Below the noise "
                   f"most likely; a full GNN-pretrain build is hard to justify.")
    else:
        verdict = (f"RED: no homo-transfer gain at current scale ({gain:+.3f} kcal). Confirms "
                   f"'open problem 1' null extends to ~23k cross / 30k homo. Lever closed.")
    out = {"built_by": "cross_benzoin/homo_cross_joint_tabular.py",
           "n_homo": len(homo), "n_cross_train": len(xtr), "n_cross_holdout": len(xte),
           "features": "72 shared (34 prod QM + 22 donor-side ald QM + 16 RDKit global) + is_homo",
           "results": res, "cross_reference_full_model": CROSS_REF,
           "verdict": verdict, "runtime_min": round((time.time() - t0) / 60, 1)}
    (OUT / "homo_cross_joint_tabular.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n" + verdict)
    print(f"-> {OUT / 'homo_cross_joint_tabular.json'}  ({out['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
