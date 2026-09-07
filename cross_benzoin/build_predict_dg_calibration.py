#!/usr/bin/env python
"""Build the calibration artifact `predict_dg_calibration.json` that
`predict_dg.py` uses to emit (a) split-conformal prediction intervals and
(b) a g-xTB-baseline-failure risk flag for heteroatom-heavy pairs.

Rationale
---------
* The point estimate `dG_pred_kcal` is capped by a ~2.9 kcal single-conformer
  DFT label-noise floor (confnoise_cross). `ens_member_sigma` (spread of the 3
  base learners) is a *directional* uncertainty only -- it has never been
  calibrated to an actual coverage guarantee. Split-conformal fixes that:
  a distribution-free interval with marginal coverage >= 1-alpha, calibrated on
  held-out (scaffold-disjoint) residuals.
* The homo dG hard-tail analysis (homo_active_relabel_null_result memory,
  pipeline/bde/STATUS.md S3.6) found the worst residuals are a *g-xTB baseline
  failure*, not a model/label gap: on the 150 worst P / sulfonyl / imine /
  amide molecules |g-xTB baseline error| averaged 13.9 kcal (vs 4.3 global) and
  corr(residual, baseline error) = 0.888. Those substructures are identifiable
  up front by SMARTS -- so we can warn the user "for this pair the cheap
  baseline is unreliable, route it to DFT" instead of silently returning a
  number the model can't get right.

This script runs locally (nequip env, has torch_geometric for the GNN):

  /home/schen3/venv/nequip/bin/python cross_benzoin/build_predict_dg_calibration.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cross_benzoin"))
from predict_cross_champion import CrossBenzoinBlendPredictor  # noqa: E402

TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
MODEL_DIR = REPO / "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1"
GNN_DIR = REPO / "data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1"
OUT = REPO / "cross_benzoin/predict_dg_calibration.json"

# g-xTB-baseline-failure motifs. Each entry: name -> SMARTS. Evidence trail in
# the module docstring; per-motif holdout MAE is written into the JSON so the
# deployment flag is auditable, not just asserted.
RISK_SMARTS = {
    "hypervalent_P": "[#15]=[#8]",                                  # phosphine oxide / phosphonate / phosphate
    "sulfonyl": "[#16X4](=[#8])(=[#8])",                            # -S(=O)(=O)-
    "sulfoxide": "[#16X3;$([#16X3]=[#8])]",                         # -S(=O)- (not sulfonyl)
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "N_oxide": "[$([#7X4+][#8X1-]),$([#7X3+][#8X1-]),$([nX3+][#8X1-])]",
    "selenium": "[#34]",
    "triflate_like": "[#6](F)(F)(F)[#16X4](=[#8])(=[#8])",         # -CF3 on a sulfonyl (triflate / triflone)
}
_RISK_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in RISK_SMARTS.items()}
assert all(p is not None for p in _RISK_PATTERNS.values()), "bad SMARTS"


def risk_motifs_for_smiles(smis: list[str]) -> list[str]:
    """Union of matched risk-motif names across a set of SMILES (donor,
    acceptor, product)."""
    hits: set[str] = set()
    for smi in smis:
        if not isinstance(smi, str) or not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for name, patt in _RISK_PATTERNS.items():
            if m.HasSubstructMatch(patt):
                hits.add(name)
    return sorted(hits)


def split_conformal_q(resid_abs: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of |residual|."""
    n = len(resid_abs)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(resid_abs)[k - 1])


def coverage(y_true, y_pred, lo, hi):
    return float(np.mean((y_true >= y_pred - lo) & (y_true <= y_pred + hi))) if False else \
        float(np.mean((y_true >= lo) & (y_true <= hi)))


def main() -> int:
    df = pd.read_parquet(TABLE)
    pred = CrossBenzoinBlendPredictor.load(str(MODEL_DIR), gnn_dir=str(GNN_DIR))

    parts = {}
    for split in ("test", "validation", "train"):
        sub = df[df["new_scaffold_split"] == split].reset_index(drop=True)
        if sub.empty:
            continue
        yhat = np.asarray(pred.predict(sub), dtype=float)
        y = sub["dG_orca_kcal"].to_numpy(dtype=float)
        # same cheap 3-learner spread predict_dg.py emits as ens_member_sigma
        ens = pred.ensemble
        X = sub[ens.feats].apply(pd.to_numeric, errors="coerce").fillna(ens.medians).fillna(0.0)
        members = np.vstack([ens.mlp.predict(ens.scaler.transform(X)),
                             ens.xgb_a.predict(X), ens.xgb_b.predict(X)])
        sigma = members.std(axis=0)
        gxtb = sub["dG_gxtb_kcal"].to_numpy(dtype=float)
        motifs = [risk_motifs_for_smiles([sub.at[i, "donor_smiles"], sub.at[i, "acceptor_smiles"],
                                          sub.at[i, "smiles"]]) for i in range(len(sub))]
        parts[split] = dict(y=y, yhat=yhat, sigma=sigma, gxtb=gxtb, motifs=motifs,
                            resid=np.abs(y - yhat))

    # --- calibrate on test + validation pooled (n ~= 929). Neither is a
    # perfectly untouched set (test drove blend w_gnn, validation may drive GNN
    # early-stop) but both are outside the XGB/MLP fit and outside training
    # scaffolds -- disclosed in the JSON. Coverage is then reported per-split as
    # the diagnostic. ---
    calib = np.concatenate([parts[s]["resid"] for s in ("test", "validation") if s in parts])
    sig_c = np.concatenate([parts[s]["sigma"] for s in ("test", "validation") if s in parts])
    lam = float(np.median(sig_c))  # regulariser for normalised nonconformity
    norm_calib = calib / (sig_c + lam)

    out = {
        "built_by": "cross_benzoin/build_predict_dg_calibration.py",
        "champion": {"model_dir": str(MODEL_DIR.relative_to(REPO)),
                     "gnn_dir": str(GNN_DIR.relative_to(REPO)), "blend_w_gnn": 0.50},
        "calibration_set": "scaffold-disjoint test + validation splits pooled",
        "n_calib": int(len(calib)),
        "caveats": ("test split was used to pick blend w_gnn and validation may "
                    "drive GNN early-stopping; both are outside the XGB/MLP fit "
                    "and outside training scaffolds. Marginal coverage only "
                    "(not conditional). Point estimate still capped by the ~2.9 "
                    "kcal single-conformer DFT label-noise floor."),
        "conformal": {},
        "sigma_lambda": lam,
        "risk_smarts": RISK_SMARTS,
    }
    for alpha in (0.10, 0.20):
        q_global = split_conformal_q(calib, alpha)
        q_norm = split_conformal_q(norm_calib, alpha)
        rec = {"alpha": alpha, "nominal_coverage": round(1 - alpha, 3),
               "q_global_kcal": q_global, "q_normalized": q_norm,
               "per_split_coverage": {}}
        for s, p in parts.items():
            if s == "train":
                continue
            g_cov = coverage(p["y"], None, p["yhat"] - q_global, p["yhat"] + q_global)
            n_lo = p["yhat"] - q_norm * (p["sigma"] + lam)
            n_hi = p["yhat"] + q_norm * (p["sigma"] + lam)
            n_cov = float(np.mean((p["y"] >= n_lo) & (p["y"] <= n_hi)))
            rec["per_split_coverage"][s] = {
                "global": round(g_cov, 3), "global_halfwidth_kcal": round(q_global, 3),
                "normalized": round(n_cov, 3),
                "normalized_median_halfwidth_kcal": round(float(np.median(q_norm * (p["sigma"] + lam))), 3),
            }
        out["conformal"][f"{int((1-alpha)*100)}pct"] = rec

    # sigma guard: the conformal interval is marginal and does NOT widen for
    # out-of-distribution structures (e.g. the thiourea / complex-fluorinated
    # pairs whose 3-learner spread blows up to 50-80 kcal). Flag those
    # explicitly so a user knows to ignore both the point estimate and the
    # interval there.
    out["sigma_guard"] = {
        "p95": round(float(np.percentile(sig_c, 95)), 3),
        "p99": round(float(np.percentile(sig_c, 99)), 3),
        "max": round(float(np.max(sig_c)), 3),
        "high_sigma_threshold_kcal": round(float(np.percentile(sig_c, 99)), 3),
        "note": "predict_dg.py sets dg_high_sigma = ens_member_sigma > high_sigma_threshold_kcal",
    }

    # --- risk flag validation on the test split ---
    tp = parts["test"]
    flagged = np.array([len(m) > 0 for m in tp["motifs"]])
    def _stats(mask):
        if mask.sum() == 0:
            return {"n": 0}
        return {"n": int(mask.sum()),
                "blend_mae": round(float(np.mean(tp["resid"][mask])), 3),
                "mean_abs_baseline_err": round(float(np.mean(np.abs(tp["gxtb"][mask] - tp["y"][mask]))), 3),
                "p90_abs_resid": round(float(np.percentile(tp["resid"][mask], 90)), 3)}
    per_motif = {}
    for name in RISK_SMARTS:
        m = np.array([name in mm for mm in tp["motifs"]])
        per_motif[name] = _stats(m)
    out["risk_flag_validation_on_test"] = {
        "flagged": _stats(flagged),
        "not_flagged": _stats(~flagged),
        "per_motif": per_motif,
        "note": ("flagged rows should show materially higher blend_mae and "
                 "mean_abs_baseline_err than not_flagged for the flag to be "
                 "evidence-backed; per_motif lets predict_dg.py / reviewers see "
                 "which motifs actually carry the signal at this holdout size."),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
