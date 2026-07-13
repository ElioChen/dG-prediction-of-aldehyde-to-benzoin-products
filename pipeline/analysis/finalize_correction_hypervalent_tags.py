#!/usr/bin/env python
"""Tier-1b of the 2026-07-10 external-diagnosis review (REPORT_review_external_diagnosis_
20260710.md Action D): add EXPLICIT hypervalent/functional-group SMARTS features (sulfonyl,
sulfonyl-F, nitro, nitrile, imine, has_B/Si/P, soft-S-thioether, halogen, ester, amide -- both
boolean presence AND match-count, both aldehyde+product side) to the champion 275-feat set and
retrain, to see whether giving the tree an explicit split feature for the hypervalent subclass
(instead of making it infer the class from continuous electronic descriptors) moves the MAE --
globally, and specifically on the sulfonyl/P/imine/amide hard subset identified in
REPORT_deep_error_analysis_champion275_20260707.md (11.2x/9.5x/3.6x/3.4x enrichment).

Same 70:20:10 split (seed 42), same ensemble (MLP+XGB_d8+XGB_d10) as
finalize_correction_mordredslim271_bdegxtb.py, so the MAE is directly comparable. Reports BOTH
overall MAE and hard-subset-only MAE for champion-275 vs champion-275+tags, matching the noise
band from REPORT_robustness_baseline72_20260702.md (1.571 +/- 0.013 on the full 72-feat run --
used here only as a rough "is this delta real" ruler, not a literal band for the hard subset).
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")

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
FEATS_72 = PROD_QM + ALDp + GLOB
BDE_COLS = ["prod_bdfe_gxtb_kcal", "ald_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]
NAME = "MORDREDSLIM271_BDEGXTB_HYPERTAGS"

# hypervalent/functional-group SMARTS -- same set as deep_error_analysis_champion275.py's
# nonewg-outlier-drivers hard-subset tags, but here fed to the MODEL, not just diagnosis
SMARTS = {
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "sulfonyl_F": "[#16X4](=[OX1])(=[OX1])F",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "[CX2]#[NX1]",
    "imine": "[CX3]=[NX2]",
    "has_B": "[#5]",
    "has_Si": "[#14]",
    "has_P": "[#15]",
    "soft_S_thioether": "[#16X2]",
    "halogen": "[F,Cl,Br,I]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
HARD_TAGS = ["sulfonyl", "has_P", "imine", "amide"]


def tag_counts(smi):
    m = Chem.MolFromSmiles(str(smi))
    out = {}
    for k, p in PATS.items():
        n = len(m.GetSubstructMatches(p)) if m is not None else 0
        out[f"tag_{k}"] = int(n > 0)
        out[f"tagcnt_{k}"] = n
    return out


TAG_COLS = [f"tag_{k}" for k in SMARTS] + [f"tagcnt_{k}" for k in SMARTS]


def gfeats(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return {f"g_{k}": np.nan for k in GKEYS}
    s = {a.GetSymbol() for a in m.GetAtoms()}
    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),
            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]
    return {f"g_{k}": v for k, v in zip(GKEYS, vals)}


def add_global(df, smi_col):
    u = df[[smi_col]].drop_duplicates()
    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values
    return df.merge(g, on=smi_col, how="left")


def add_tags(df, smi_col, prefix):
    u = df[[smi_col]].drop_duplicates()
    t = pd.DataFrame([tag_counts(s) for s in u[smi_col]]).add_prefix(prefix)
    t[smi_col] = u[smi_col].values
    return df.merge(t, on=smi_col, how="left")


def _xgb(depth, ne):
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                        early_stopping_rounds=60, eval_metric="mae")


def make_members():
    return [("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                                 early_stopping=True, n_iter_no_change=12)),
            ("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]


def run(df, feats, label, hard_mask_te=None):
    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(df[feats].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(df[feats].values[tr]), imp.transform(df[feats].values[va]), imp.transform(df[feats].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
    dtr, dva = df.delta.values[tr], df.delta.values[va]
    gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    members = make_members(); preds_te = []
    for nm, m in members:
        if nm.startswith("XGB"): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
        else: m.fit(Xtr, dtr)
        preds_te.append(m.predict(Xte))
    pred = np.vstack(preds_te).mean(0)
    yhat = gte + pred; err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())

    result = {"label": label, "n_feat": len(feats), "n": len(df), "mae": mae, "rmse": rmse, "r2": r2, "te_idx": te}
    scope = {}
    if "cls" in df.columns:
        for s in ["aromatic", "aliphatic"]:
            mk = df.cls.values[te] == s
            if mk.sum() > 50: scope[s] = float(err[mk].mean())
    result["scope"] = scope
    if hard_mask_te is not None:
        hm = hard_mask_te[te]
        result["hard_mae"] = float(err[hm].mean()) if hm.sum() > 0 else None
        result["hard_n"] = int(hm.sum())
        result["bg_mae"] = float(err[~hm].mean()) if (~hm).sum() > 0 else None
    print(f"[{label}] n_feat={len(feats)} n={len(df):,} test MAE={mae:.3f} RMSE={rmse:.3f} "
          f"R2={r2:.3f} scope={scope} hard_mae={result.get('hard_mae')} (n={result.get('hard_n')}) "
          f"bg_mae={result.get('bg_mae')}", flush=True)
    return result


def main():
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")

    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    kept_mordred = set(json.load(open(f"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json"))["kept_mordred"])
    prod_kept = [c for c in kept_mordred if not c.startswith("ald_")]
    ald_kept_raw = [c[len("ald_"):] for c in kept_mordred if c.startswith("ald_")]
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c in prod_kept]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c in ald_kept_raw]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]: prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]: ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")
    full = add_tags(full, "smiles", "prod_")
    full = add_tags(full, "ald_smiles", "ald_")
    prod_tag_cols = [f"prod_{c}" for c in TAG_COLS]
    ald_tag_cols = [f"ald_{c}" for c in TAG_COLS]

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    FEATS_BASE = FEATS_72 + prod_mrd_cols + ald_mrd_cols + BDE_COLS
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,}", flush=True)

    hard_mask = np.zeros(len(df), dtype=bool)
    for t in HARD_TAGS:
        hard_mask |= (df[f"prod_tag_{t}"].fillna(0).astype(bool).values |
                      df[f"ald_tag_{t}"].fillna(0).astype(bool).values)
    print(f"hard subset (sulfonyl/P/imine/amide): {hard_mask.sum():,}/{len(df):,} ({hard_mask.mean()*100:.1f}%)", flush=True)

    feats_champion = FEATS_BASE
    feats_plus_tags = FEATS_BASE + prod_tag_cols + ald_tag_cols

    results = {}
    for label, feats in [("champion275", feats_champion), ("champion275_plus_hypertags", feats_plus_tags)]:
        results[label] = run(df, feats, label, hard_mask_te=hard_mask)

    rep = OUT / f"REPORT_hypertags_augment_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Hypervalent-tag feature augmentation ({TAG})\n\n")
        fh.write("Tier-1b of the 2026-07-10 external-diagnosis review (Action D): adds explicit "
                 "SMARTS-based hypervalent/functional-group tags (sulfonyl, sulfonyl-F, nitro, "
                 "nitrile, imine, has_B/Si/P, soft-S-thioether, halogen, ester, amide -- boolean + "
                 "match-count, both aldehyde and product side, 24 new columns) to the champion "
                 "275-feat set. Same 70:20:10 split (seed 42), same ensemble as the production "
                 "champion.\n\n")
        fh.write("| variant | n_feat | test MAE | RMSE | R2 | hard-subset MAE (n) | background MAE |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for label, r in results.items():
            fh.write(f"| {label} | {r['n_feat']} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['r2']:.3f} | "
                     f"{r['hard_mae']:.3f} ({r['hard_n']}) | {r['bg_mae']:.3f} |\n")
        d_all = results["champion275_plus_hypertags"]["mae"] - results["champion275"]["mae"]
        d_hard = results["champion275_plus_hypertags"]["hard_mae"] - results["champion275"]["hard_mae"]
        fh.write(f"\n**Delta overall MAE: {d_all:+.3f}** (noise band on the 72-feat baseline is "
                 "+/-0.013 over reshuffled seeds -- treat |delta|<0.02-0.03 as noise, per "
                 "descriptor-search-exhausted memory). **Delta hard-subset MAE: {:+.3f}**.\n".format(d_hard))
        fh.write("\nIf the hard-subset delta is negative well beyond noise while the overall delta "
                 "is flat, the explicit tags are a real, localized win worth keeping even though "
                 "they don't move the global number (which is dominated by the 85%+ easy "
                 "majority). If both deltas are flat, Action D is also a null result, same "
                 "conclusion pattern as Action A (atom-local P_int) and ADCH/QTAIM.\n")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "te_idx"} for k, v in results.items()},
              open(OUT / f"hypertags_augment_results_{TAG}.json", "w"), indent=2)
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
