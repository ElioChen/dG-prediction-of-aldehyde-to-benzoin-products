#!/usr/bin/env python
"""
Single deployable inference artifact for the BDE-prediction project's champion, B6
(D-MPNN + local-3D-descriptor hybrid, see PROGRESS_20260714.md section O-10/O-14 and memory
bde-prediction-project-takeover-20260717.md). Mirrors cross_benzoin/predict_cross_champion.py's
pattern for the sibling dG project.

Two separate models exist (aldehyde formyl C-H BDE, product ketC-carbC BDE) -- both trained
via submit_b6_scaffold_disjoint_ckpt.sh (job 24706282, 2026-07-17), the checkpoint-saving
rerun of the confirmed honest-scaffold-disjoint champion (job 24694779: aldehyde MAE
1.579/R^2 0.843, product MAE 3.060/R^2 0.886; the checkpoint rerun reproduced 1.591/0.838 and
3.032/0.888, consistent within normal seed variance). Checkpoints bundle everything needed
for inference (model_state_dict, x_scaler, y_scaler, feats, params) -- see
train_gnn_hybrid_bde.py's --save-checkpoint block.

Input: a DataFrame with `smiles` + the model's named local-3D descriptor columns (see
`.feats` after loading, or LOCAL_FEATURES["aldehydes"/"products"] in train_gnn_hybrid_bde.py)
-- i.e. the standard output of this project's xTB/g-xTB + Multiwfn ADCH/QTAIM featurization
pipeline, NOT raw unfeaturized SMILES (same convention as predict_cross_champion.py and every
other model in this project -- a true end-to-end "SMILES in" entry point would need to run
fresh xTB/Multiwfn compute per molecule and was explicitly out of scope here, same as it was
for the cross-benzoin predictor).

Usage (as a library):
    from predict_bde_champion import BDEChampionPredictor
    ald_pred = BDEChampionPredictor.load("aldehydes")
    bde_kcal = ald_pred.predict(df)   # np.ndarray, kcal/mol

Usage (CLI smoke test, reproduces the checkpoint's own held-out test MAE):
    python pipeline/bde/predict_bde_champion.py --which aldehydes --verify
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn_hybrid_bde import build_mpnn, make_dataset, predict as _cp_predict

REPO_DATA = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
CKPT_PATHS = {
    "aldehydes": Path("/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/"
                       "models/b6_aldehydes_scaffold_disjoint.pt"),
    "products": Path("/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/"
                      "models/b6_products_scaffold_disjoint.pt"),
}
TEST_TABLES = {
    "aldehydes": REPO_DATA / "aldehydes_scaffold_split_from_dG.csv",
    "products": REPO_DATA / "products_scaffold_split.csv",
}


@dataclass
class BDEChampionPredictor:
    which: str
    model: object
    x_scaler: object
    y_scaler: object
    feats: list[str]
    params: dict

    @classmethod
    def load(cls, which: str, ckpt_path: str | Path | None = None) -> "BDEChampionPredictor":
        assert which in ("aldehydes", "products")
        ckpt_path = Path(ckpt_path) if ckpt_path is not None else CKPT_PATHS[which]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_mpnn(ckpt["n_xd"], ckpt["params"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[BDEChampionPredictor] loaded {which} checkpoint from {ckpt_path} "
              f"(training result: MAE={ckpt['result']['MAE']:.3f} R2={ckpt['result']['R2']:.3f})")
        return cls(which=which, model=model, x_scaler=ckpt["x_scaler"], y_scaler=ckpt["y_scaler"],
                    feats=ckpt["feats"], params=ckpt["params"])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns predicted BDE (kcal/mol) for each row of df."""
        from chemprop import featurizers
        Xraw = df[self.feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Xs = self.x_scaler.transform(Xraw)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dset = make_dataset(df["smiles"].to_numpy(), Xs, None, featurizer)

        from lightning import pytorch as pl
        trainer = pl.Trainer(accelerator="auto", devices=1, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
        pred_sc = _cp_predict(self.model, trainer, dset, self.params)
        return self.y_scaler.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--table", default=None, help="defaults to the champion's own scaffold-disjoint test rows")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--verify", action="store_true",
                     help="score the FULL held-out test split and compare against the checkpoint's own reported MAE")
    args = ap.parse_args()

    predictor = BDEChampionPredictor.load(args.which)

    from train_gnn_hybrid_bde import LOCAL_FEATURES
    from qc import qc_filter
    feats = LOCAL_FEATURES[args.which]
    id_cols = ["id", "smiles", "error"] + feats if args.which == "aldehydes" \
        else ["id", "donor_id", "smiles", "error"] + feats
    mol = pd.read_csv(REPO_DATA / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False, low_memory=False)
    mol = mol[mol["error"] == ""]
    for c in feats:
        mol[c] = pd.to_numeric(mol[c], errors="coerce")
    labels = pd.read_csv(REPO_DATA / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    labels = labels.dropna(subset=["bde_gxtb_kcal"]).drop_duplicates("id")
    # must match train_gnn_hybrid_bde.py's main() exactly (qc_filter drops physically
    # implausible / MAD-outlier labels) -- omitting this was a real bug caught during
    # verification (inflated MAE 4.47 vs the checkpoint's real 1.591, n=21681 vs 21086).
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
    df = labels.merge(mol, on="id", how="inner").dropna(subset=feats, how="all")
    df = df[df["smiles"] != ""].reset_index(drop=True)

    sf = pd.read_csv(TEST_TABLES[args.which])[["id", "scaffold_split"]]
    sf["id"] = sf["id"].astype(str)
    df = df.merge(sf, on="id", how="inner")
    test_df = df[df["scaffold_split"] == "test"].reset_index(drop=True)

    if not args.verify:
        test_df = test_df.head(args.n)

    pred = predictor.predict(test_df)
    actual = test_df["bde_gxtb_kcal"].to_numpy()
    if args.verify:
        mae = np.abs(pred - actual).mean()
        print(f"\nFULL held-out test (n={len(test_df)}) MAE = {mae:.4f} kcal/mol "
              f"(checkpoint's own reported MAE was logged at load time above -- should match)")
    else:
        for i, (p, a) in enumerate(zip(pred, actual)):
            print(f"row {i}: pred={p:.3f}  actual={a:.3f}  err={abs(p - a):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
