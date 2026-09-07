#!/usr/bin/env python
"""End-to-end cross-benzoin ΔG prediction for new aldehyde pairs.

Two halves:

  A. FEATURIZE (slow, xTB geometry -- run as a SLURM job, submit_predict_dg.sh):
       pairs CSV -> cb_featurize.py --pairs -> product QM/xTB descriptors + g-xTB SP
     Produces  <workdir>/features.csv .  Skip with --products-csv if you already
     have a cross_round*_dft_products.csv-schema file for the pairs.

  B. ASSEMBLE + PREDICT (fast, this script's default):
       products table  (+ donor_*/acceptor_* aldehyde descriptors pulled from the
       220k library by canonical SMILES, + product 2D mordred, + interaction_*,
       + dG_gxtb baseline)  -> prune to the frozen 260 champion features
       -> CrossBenzoinBlendPredictor.predict  ->  dG_orca_kcal + ensemble σ.

Output columns: dG_pred_kcal (point estimate, MAE 2.215 on holdout, capped by a ~2.9
kcal/mol single-conformer DFT label-noise floor -- see confnoise_cross), ens_member_sigma
(cheap directional uncertainty), and three Goal-3 reformulation columns that are NOT
capped by that floor (validated AUC 0.92-0.93, see eval_reformulation_classification_
ranking.py): dg_favorable (dG<0, the physically meaningful cut), dg_below_train_median,
dg_rank_pct (within-batch percentile, 0=most favorable -- only meaningful when scoring a
batch of candidates against each other).

Aldehydes MUST already be in data/library (checked by canonical SMILES); truly
novel aldehydes need their own cb_featurize --emit-aldehydes pass first (not yet
wired here). NOTE (2026-09-07): donor_G_gxtb/acceptor_G_gxtb (2/260 frozen features) are
median-imputed for ~207k/209k library aldehydes post the 09-06 BDE rebuild, which never
recomputes that whole-molecule quantity -- see [[predict-dg-g-gxtb-regression-fixed]] in
Claude memory / RUN_LOG 09-07. Bounded, disclosed quality cost, not a correctness bug.

  # pairs already have a products table (e.g. a slice of cross_round10_products_merged.csv):
  python cross_benzoin/predict_dg.py --products-csv pairs_products.csv --out preds.csv

  # from scratch: first sbatch submit_predict_dg.sh <pairs.csv> <workdir>, then:
  python cross_benzoin/predict_dg.py --products-csv <workdir>/products_for_assemble.csv --out preds.csv
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cross_benzoin"))
sys.path.insert(0, str(REPO / "pipeline"))
# run THIS script under nequip (torch_geometric for the GNN); shell out to
# nhc-workflow for the mordred / rdkit / assemble steps (nequip lacks mordred).
FEAT_PY = "/home/schen3/venv/nhc-workflow/bin/python"
SCHEMA = REPO / "data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json"
CHAMP = REPO / "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1"
GNN = REPO / "data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1"


def _stage_fake_round(products_csv: Path, tmp: Path) -> None:
    """Put the products table + a dummy dft_sp + product mordred where
    assemble_cross_training_table_v3.round_paths(99) expects them."""
    rdir = REPO / "data/cross_benzoin/cross_round99"
    ddir = REPO / "data/raw/dft_sp_cross/cross_round99"
    rdir.mkdir(parents=True, exist_ok=True); ddir.mkdir(parents=True, exist_ok=True)
    prod = pd.read_csv(products_csv, low_memory=False)
    assert "id" in prod.columns, "products csv needs an 'id' column (product InChIKey__InChIKey)"
    prod.to_csv(rdir / "cross_round99_dft_products.csv", index=False)
    # dummy label so load_round's inner-join keeps every row (predict ignores the target)
    pd.DataFrame({"id": prod["id"], "dG_orca_kcal": 0.0}).to_csv(
        ddir / "cross_round99_dft_sp.csv", index=False)
    # product g-xTB BDE (bde_gxtb_kcal is one of the frozen 260 feats): pull rows for
    # these ids from any already-computed round's bde_gxtb/, NaN for genuinely new pairs
    # (median-filled at prune; a real featurize pass would emit it).
    bdir = rdir / "bde_gxtb"; bdir.mkdir(exist_ok=True)
    have = []
    for f in sorted((REPO / "data/cross_benzoin").glob("cross_round*/bde_gxtb/chunk_*.csv")):
        try:
            c = pd.read_csv(f, usecols=lambda x: x in ("id", "bde_gxtb_kcal"))
            have.append(c[c["id"].isin(prod["id"])])
        except Exception:
            pass
    bde = (pd.concat(have, ignore_index=True).drop_duplicates("id")
           if have else pd.DataFrame(columns=["id", "bde_gxtb_kcal"]))
    miss = prod.loc[~prod["id"].isin(bde["id"]), ["id"]].assign(bde_gxtb_kcal=float("nan"))
    pd.concat([bde, miss], ignore_index=True).to_csv(bdir / "chunk_0000.csv", index=False)
    # product mordred
    mdir = tmp / "mordred_products"; mdir.mkdir(exist_ok=True)
    n = len(prod)
    for ci in range((n + 99) // 100):
        subprocess.run([FEAT_PY, str(REPO / "cross_benzoin/add_mordred_cross_products.py"),
                        "--products-csv", str(rdir / "cross_round99_dft_products.csv"),
                        "--chunk-id", str(ci), "--chunk-size", "100", "--out-dir", str(mdir)],
                       check=True)
    pd.concat([pd.read_csv(f) for f in sorted(mdir.glob("chunk_*.csv"))], ignore_index=True) \
      .drop_duplicates("id").to_csv(rdir / "round99_products_mordred.csv", index=False)


# Goal-3 reformulation (validated 2026-09-04, see
# cross_benzoin/eval_reformulation_classification_ranking.py and
# data/cross_benzoin/reformulation_classification_ranking_eval.json): pinpoint kcal/mol
# MAE is capped by the ~2.9 kcal single-conformer DFT label-noise floor, but the SAME
# champion's favorable/unfavorable classification and top-k ranking are excellent and
# robust to that floor by construction (AUC 0.92-0.93, top-10% precision 0.644 vs g-xTB's
# 0.311). T=0 is the physically meaningful cut (thermodynamically favorable reaction);
# T=TRAIN_MEDIAN_DG_KCAL is the higher-recall balanced-class cut, both scored in the eval.
FAVORABLE_THRESHOLD_KCAL = 0.0
TRAIN_MEDIAN_DG_KCAL = 4.917011302989063  # r1-10 train split median dG_orca_kcal, frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-csv", required=True, type=Path,
                    help="cross_round*_dft_products.csv-schema file for the pairs to predict")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-dir", default=str(CHAMP))
    ap.add_argument("--gnn-dir", default=str(GNN))
    args = ap.parse_args()

    r99 = REPO / "data/cross_benzoin/cross_round99"
    d99 = REPO / "data/raw/dft_sp_cross/cross_round99"
    with tempfile.TemporaryDirectory(dir=REPO / "data/cross_benzoin") as td:
      tmp = Path(td)
      try:
        _stage_fake_round(args.products_csv, tmp)

        subprocess.run([FEAT_PY, str(REPO / "cross_benzoin/assemble_cross_training_table_v3.py"),
                        "--rounds", "99", "--out-tag", "predict_tmp",
                        "--product-mordred-csv",
                        str(REPO / "data/cross_benzoin/cross_round99/round99_products_mordred.csv")],
                       check=True)
        tbl = REPO / "data/cross_benzoin/cross_round99/cross_train_table_predict_tmp.parquet"

        slim = tmp / "slim.parquet"
        subprocess.run([FEAT_PY, str(REPO / "cross_benzoin/prune_table_to_champion_features.py"),
                        "--table", str(tbl), "--feature-list", str(SCHEMA), "--out", str(slim)],
                       check=True)

        from predict_cross_champion import CrossBenzoinBlendPredictor  # noqa: E402
        df = pd.read_parquet(slim)
        pred = CrossBenzoinBlendPredictor.load(args.model_dir, gnn_dir=args.gnn_dir)
        dg = pred.predict(df)
        # cheap uncertainty proxy: spread of the 3 base learners (MLP, XGB-a, XGB-b).
        # NOT the full pair-grouped bootstrap epistemic estimate (score_round_active_
        # learning.py --n-boot) -- fast and directional only.
        ens = pred.ensemble
        X = df[ens.feats].apply(pd.to_numeric, errors="coerce").fillna(ens.medians).fillna(0.0)
        members = np.vstack([ens.mlp.predict(ens.scaler.transform(X)),
                             ens.xgb_a.predict(X), ens.xgb_b.predict(X)])
        sigma = members.std(axis=0)

        out = df[["id", "donor_id", "acceptor_id", "smiles", "dG_gxtb_kcal"]].copy()
        out["dG_pred_kcal"] = dg
        out["ens_member_sigma"] = sigma
        # Goal-3 reformulation columns (see FAVORABLE_THRESHOLD_KCAL above): a
        # favorable/unfavorable screen and a within-batch percentile rank, both
        # validated as robust to the label-noise floor that caps the raw kcal MAE.
        out["dg_favorable"] = out["dG_pred_kcal"] < FAVORABLE_THRESHOLD_KCAL
        out["dg_below_train_median"] = out["dG_pred_kcal"] < TRAIN_MEDIAN_DG_KCAL
        # rank 0 = most favorable (most negative dG) in this batch; only meaningful
        # for screening/ranking a batch of candidates against each other, not a
        # single pair.
        out["dg_rank_pct"] = out["dG_pred_kcal"].rank(pct=True, method="average")
        out.to_csv(args.out, index=False)
        print(out.to_string(index=False))
        n_fav = int(out["dg_favorable"].sum())
        print(f"\n{n_fav}/{len(out)} pairs predicted favorable (dG < {FAVORABLE_THRESHOLD_KCAL} kcal/mol). "
              f"See eval_reformulation_classification_ranking.py for this screen's validated "
              f"AUC/precision (reformulation_classification_ranking_eval.json) -- the kcal/mol "
              f"point estimate is capped by ~2.9 kcal label noise, this favorable/unfavorable "
              f"call and the dg_rank_pct ranking are not.")
      finally:
        shutil.rmtree(r99, ignore_errors=True)
        shutil.rmtree(d99, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
