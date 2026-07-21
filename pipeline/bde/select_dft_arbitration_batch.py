#!/usr/bin/env python
"""Active-learning-guided selection for the NEXT DFT arbitration batch (2026-07-16/17).

Replaces the manual "3 molecules per functional group, spread across the g-xTB BDE range"
recipe `dft_bde_geom_arbitration.py` used for its n=18 pilot with the same
bootstrap-ensemble-uncertainty active-learning recipe cross_benzoin's
`score_round_active_learning.py` already validated across 6 rounds -- just retargeted from
"which pair is most useful to train the dG delta-model" to "which aldehyde is most useful to
send to expensive DFT arbitration next."

Honest caveat: we only have n=17 real (g-xTB, DFT) comparison points so far (job 24665669),
several functional groups with as few as n=2-3. A supervised uncertainty model trained on
17 points is a WEAK signal -- it is used here only as a tie-breaker on top of two much more
load-bearing components: (1) functional-group stratification (reusing the residual-analysis
groups, now including selenium -- the single worst offender at 3.7x, which the n=18 pilot's
TARGET_GROUPS omitted) and (2) farthest-point diversity sampling in descriptor space within
each group (the n=18 pilot only diversified along the 1-D g-xTB BDE value; this diversifies
across the actual QM descriptor space so the next batch doesn't cluster on near-duplicates).

This script only SELECTS candidates -- it does not submit any DFT compute. Each DFT
arbitration data point costs hours (job 24665669's shards ranged ~2-10h for 2 molecules
each), so the resulting candidate list should be reviewed before committing to a batch size.

Usage:
  python select_dft_arbitration_batch.py --per-group 25 --out /tmp/dft_arb_round2_candidates.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
ARB_DIR = H / "dft_bde_geom_arbitration"

TARGET_GROUPS = [
    ("selenium", "[#34]"),
    ("sulfonyl", "[#16X4](=[OX1])(=[OX1])"),
    ("imine", "[CX3]=[NX2]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("n_oxide", "[#7+][#8X1-]"),
    ("phosphorus", "[#15]"),
]
CONTROL_GROUPS = [("halogen_plain", "[F,Cl,Br,I]")]

DIVERSITY_FEATS = ["xtb_eta", "xtb_EA", "xtb_dipole", "SASA_total",
                   "sterimol_L", "sterimol_B1", "sterimol_B5"]
UNCERTAINTY_FEATS = DIVERSITY_FEATS + ["xtb_HOMO", "xtb_LUMO", "P_int"]


def tag_first_group(smi, pats):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    for lab, p in pats:
        if p is not None and m.HasSubstructMatch(p):
            return lab
    return None


def farthest_point_sample(X: np.ndarray, k: int, seed: int) -> list[int]:
    """Greedy farthest-point sampling: start from a random point, repeatedly add the
    point farthest (in min-distance-to-selected-set) from what's already picked."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n <= k:
        return list(range(n))
    picked = [int(rng.integers(n))]
    dmin = np.linalg.norm(X - X[picked[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dmin))
        picked.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=25,
                     help="target molecules per functional group (PROGRESS_20260714.md's "
                          "own stated target for a conclusive per-class noise-floor read)")
    ap.add_argument("--shortlist-mult", type=int, default=4,
                     help="diversity shortlist size = per-group * this, before uncertainty re-rank")
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ald_all = pd.read_csv(H / "aldehydes_all.csv",
                           usecols=["id", "smiles", "error"] + UNCERTAINTY_FEATS,
                           dtype={"id": str}, keep_default_na=False, low_memory=False)
    ald_all = ald_all[ald_all["error"] == ""].drop_duplicates("id")
    for c in UNCERTAINTY_FEATS:
        ald_all[c] = pd.to_numeric(ald_all[c], errors="coerce")
    # ids are stored inconsistently across files ("4199" vs "4199.0" style) -- normalize
    # to a plain int-string everywhere in this script to avoid silent merge/exclusion misses
    ald_all["id"] = ald_all["id"].astype(float).astype(int).astype(str)
    labels = pd.read_csv(H / "aldehydes_bdfe_gxtb_descriptors.csv", dtype={"id": str}) \
        .dropna(subset=["bde_gxtb_kcal"])
    labels["id"] = labels["id"].astype(float).astype(int).astype(str)
    df = ald_all.merge(labels[["id", "bde_gxtb_kcal"]], on="id", how="inner")
    for c in UNCERTAINTY_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=UNCERTAINTY_FEATS).reset_index(drop=True)
    print(f"candidate pool before exclusions: {len(df)} aldehydes", flush=True)

    # exclude the 18 already sent to DFT arbitration
    shards = sorted(glob.glob(str(ARB_DIR / "shard_*.csv")))
    done = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True) if shards else pd.DataFrame()
    if len(done):
        done["id_str"] = done["id"].astype(float).astype(int).astype(str)
        df = df[~df["id"].isin(done["id_str"])].reset_index(drop=True)
    print(f"candidate pool after excluding {len(done)} already-arbitrated: {len(df)}", flush=True)

    pats_t = [(lab, Chem.MolFromSmarts(sm)) for lab, sm in TARGET_GROUPS]
    pats_c = [(lab, Chem.MolFromSmarts(sm)) for lab, sm in CONTROL_GROUPS]
    df["grp"] = df["smiles"].map(lambda s: tag_first_group(s, pats_t))
    ctrl_mask = df["grp"].isna() & df["smiles"].map(lambda s: tag_first_group(s, pats_c) is not None)
    df.loc[ctrl_mask, "grp"] = "control"

    # --- weak uncertainty signal from the n=17 existing arbitration points ---
    ens_models = None
    if len(done):
        ok = done[done["error"].isna()].copy()
        ok["id_str"] = ok["id"].astype(float).astype(int).astype(str)
        ok = ok.merge(ald_all[["id"] + UNCERTAINTY_FEATS].rename(columns={"id": "id_str"}),
                      on="id_str", how="inner")
        for c in UNCERTAINTY_FEATS:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")
        ok = ok.dropna(subset=UNCERTAINTY_FEATS + ["Delta_tot"])
        print(f"training weak uncertainty ensemble on n={len(ok)} known arbitration points "
              f"(honest caveat: this is a soft tie-breaker, not a reliable model)", flush=True)
        if len(ok) >= 8:
            sc = StandardScaler().fit(ok[UNCERTAINTY_FEATS])
            Xk = sc.transform(ok[UNCERTAINTY_FEATS])
            yk = ok["Delta_tot"].abs().to_numpy()
            rng = np.random.default_rng(args.seed)
            ens_models = []
            for b in range(args.n_boot):
                idx = rng.integers(0, len(Xk), len(Xk))
                m = GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                               learning_rate=0.1, random_state=args.seed + b)
                m.fit(Xk[idx], yk[idx])
                ens_models.append(m)
            ens_scaler = sc
        else:
            print("  too few points for even a weak ensemble -- skipping uncertainty re-rank "
                  "entirely, diversity sampling only", flush=True)

    picks = []
    for lab, _ in TARGET_GROUPS + [("control", None)]:
        sub = df[df["grp"] == lab].reset_index(drop=True)
        if len(sub) == 0:
            print(f"WARN: no candidates for group {lab}", flush=True)
            continue
        Xd = StandardScaler().fit_transform(sub[DIVERSITY_FEATS])
        shortlist_n = min(len(sub), args.per_group * args.shortlist_mult)
        short_idx = farthest_point_sample(Xd, shortlist_n, args.seed)
        short = sub.iloc[short_idx].reset_index(drop=True)

        if ens_models is not None:
            Xs = ens_scaler.transform(short[UNCERTAINTY_FEATS])
            preds = np.stack([m.predict(Xs) for m in ens_models])
            short["pred_abs_delta_mean"] = preds.mean(axis=0)
            short["pred_abs_delta_std"] = preds.std(axis=0)
            # acquisition: expected |label error| + epistemic uncertainty (UCB-style)
            short["acq_score"] = short["pred_abs_delta_mean"] + short["pred_abs_delta_std"]
            short = short.sort_values("acq_score", ascending=False)
        n_take = min(args.per_group, len(short))
        picks.append(short.head(n_take).assign(sample_group=lab))
        print(f"{lab}: pool={len(sub)} shortlist={len(short)} selected={n_take}", flush=True)

    out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} candidates -> {args.out}")
    print(out["sample_group"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
