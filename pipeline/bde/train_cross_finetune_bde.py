#!/usr/bin/env python
"""Phase-1 next-step #2 (2026-07-15): does a little real cross-benzoin data beat homo-only
zero-shot transfer for product BDE prediction? PROGRESS_20260714.md section "三.五" showed
H-SPOC trained on 208k homo products generalizes to real cross products (donor != acceptor)
with real-but-degraded accuracy (R^2 0.740 in-domain -> 0.381 zero-shot). This script runs
the three follow-up variants requested there, all evaluated on the SAME held-out cross test
set for a fair comparison:

  A. cross-only    -- train H-SPOC from scratch on cross_train rows only.
  B. homo+cross     -- concatenate homo_train (208k) with cross_train, train jointly.
  C. homo->cross ft -- pretrain on homo_train, continue boosting on cross_train
                        (XGBoost `xgb_model=` warm start -- gradient boosting's analogue of
                        fine-tuning: new trees correct the residual of the frozen ones).
  (zero-shot, i.e. homo-only model evaluated with no cross training data, is also reported
  on the identical cross test split for a clean apples-to-apples baseline.)

Only the 35 local descriptors present on BOTH sides can be used (the cross training table
has no ADCH/QTAIM columns -- see DESCRIPTOR_POLICY_CROSS.md and PROGRESS_20260714.md "三.五").

Usage:
  python train_cross_finetune_bde.py --out /tmp/cross_finetune_bde.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from qc import qc_filter
from splits import molecule_cold_split, donor_acceptor_cold_split

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
CROSS_TABLE = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_round3/"
                    "cross_train_table_3rounds.csv")

GLOBAL_XTB = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta",
              "xtb_omega", "xtb_dipole"]

# Shared subset of products' LOCAL_FEATURES (train_local3d_baseline.py) that exists,
# unprefixed, as product-level columns in the cross training table (no ADCH/QTAIM there).
SHARED_FEATURES = GLOBAL_XTB + [
    "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
    "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC", "fukui_minus_ketC",
    "fukui_0_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC",
    "dual_carbC", "pa_ketO", "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1",
    "sterimol_B5", "SASA_total", "P_int", "hb_dist", "hb_angle", "dih_core",
]
assert len(SHARED_FEATURES) == 36, len(SHARED_FEATURES)

XGB_PARAMS = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, n_jobs=-1)


def load_homo(seed: int, test_frac: float):
    ald = pd.read_csv(H / "products_all.csv",
                       usecols=["id", "donor_id", "error"] + SHARED_FEATURES,
                       dtype={"id": str, "donor_id": str}, keep_default_na=False,
                       low_memory=False)
    ald = ald[ald["error"] == ""]
    for c in SHARED_FEATURES:
        ald[c] = pd.to_numeric(ald[c], errors="coerce")

    labels = pd.read_csv(H / "products_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    labels = labels.dropna(subset=["bde_gxtb_kcal"]).drop_duplicates("id")
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]

    df = labels.merge(ald, on="id", how="inner").dropna(subset=SHARED_FEATURES, how="all")
    split = molecule_cold_split(df["donor_id"], test_frac=test_frac, seed=seed)
    return df, split


def load_cross(seed: int, test_frac: float, cross_table: Path = CROSS_TABLE):
    usecols = ["donor_id", "acceptor_id", "bde_gxtb_kcal"] + SHARED_FEATURES
    if str(cross_table).endswith(".parquet"):
        # newer round4/round5 tables from assemble_cross_training_table_v3.py are parquet
        # and carry ~570 columns; read all then subset (pyarrow has no cheap usecols).
        df = pd.read_parquet(cross_table)[usecols].copy()
        df["donor_id"] = df["donor_id"].astype(str)
        df["acceptor_id"] = df["acceptor_id"].astype(str)
    else:
        df = pd.read_csv(cross_table, usecols=usecols,
                          dtype={"donor_id": str, "acceptor_id": str}, low_memory=False)
    for c in SHARED_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bde_gxtb_kcal"])
    df = df[qc_filter(df["bde_gxtb_kcal"])]
    df = df.dropna(subset=SHARED_FEATURES, how="all")
    split = donor_acceptor_cold_split(df, test_frac=test_frac, seed=seed)
    return df, split


def xy(df, mask, target="bde_gxtb_kcal"):
    sub = df[mask]
    X = sub[SHARED_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    return X, y


def evaluate(y_true, y_pred, y_train_mean):
    mean_pred = np.full(len(y_true), y_train_mean)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(spearmanr(y_true, y_pred).correlation),
        "mean_baseline_MAE": float(mean_absolute_error(y_true, mean_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--homo-test-frac", type=float, default=0.15)
    ap.add_argument("--cross-test-frac", type=float, default=0.2)
    ap.add_argument("--cross-table", type=Path, default=CROSS_TABLE,
                    help="cross training table (csv or parquet); default = round3 2322-row table. "
                         "Point at a bigger round4/round5 table to test whether more cross data "
                         "changes the A/B/C ranking.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    homo_df, homo_split = load_homo(args.seed, args.homo_test_frac)
    homo_tr = (homo_split == "train").to_numpy()
    Xh_tr, yh_tr = xy(homo_df, homo_tr)
    print(f"homo: n={len(homo_df)} train={homo_tr.sum()}", flush=True)

    cross_df, cross_split = load_cross(args.seed, args.cross_test_frac, args.cross_table)
    cross_tr = (cross_split == "train").to_numpy()
    cross_te = ~cross_tr  # test_new_donor + test_new_acceptor + test_both_new, combined
    Xc_tr, yc_tr = xy(cross_df, cross_tr)
    Xc_te, yc_te = xy(cross_df, cross_te)
    print(f"cross: n={len(cross_df)} train={cross_tr.sum()} test={cross_te.sum()} "
          f"({cross_split[cross_te].value_counts().to_dict()})", flush=True)

    results = {"n_homo_train": int(homo_tr.sum()), "n_cross_train": int(cross_tr.sum()),
               "n_cross_test": int(cross_te.sum()),
               "cross_test_bucket_counts": cross_split[cross_te].value_counts().to_dict()}

    # zero-shot: homo-only model, no cross training data at all, scored on the SAME
    # cross test split used below (apples-to-apples with A/B/C, unlike the 三.五 baseline
    # which scored zero-shot on all 4120 cross rows rather than a matched held-out subset).
    m_zeroshot = XGBRegressor(random_state=args.seed, **XGB_PARAMS).fit(Xh_tr, yh_tr)
    results["zero_shot_homo_only"] = evaluate(yc_te, m_zeroshot.predict(Xc_te), yh_tr.mean())

    # A. cross-only
    m_cross = XGBRegressor(random_state=args.seed, **XGB_PARAMS).fit(Xc_tr, yc_tr)
    results["A_cross_only"] = evaluate(yc_te, m_cross.predict(Xc_te), yc_tr.mean())

    # B. homo+cross joint
    X_joint = np.concatenate([Xh_tr, Xc_tr]); y_joint = np.concatenate([yh_tr, yc_tr])
    m_joint = XGBRegressor(random_state=args.seed, **XGB_PARAMS).fit(X_joint, y_joint)
    results["B_homo_plus_cross_joint"] = evaluate(yc_te, m_joint.predict(Xc_te), y_joint.mean())

    # C. homo-pretrained, then continue boosting on cross (fine-tune)
    m_ft = XGBRegressor(random_state=args.seed, **XGB_PARAMS).fit(
        Xc_tr, yc_tr, xgb_model=m_zeroshot.get_booster())
    results["C_homo_pretrain_cross_finetune"] = evaluate(
        yc_te, m_ft.predict(Xc_te), yc_tr.mean())

    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
