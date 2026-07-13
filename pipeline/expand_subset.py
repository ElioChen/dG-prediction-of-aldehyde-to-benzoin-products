#!/usr/bin/env python3
"""
A+ — INCREMENTAL subset expansion by MaxMin diversity (Tanimoto).

Why not more PCA+KMeans+silhouette: Morgan fingerprints are sparse binary, so
PCA-Euclidean misrepresents chemical (Tanimoto) similarity, and molecular libraries
have no natural clusters — silhouette sits near 0, so "optimal k" is illusory.
MaxMin picks a maximally-diverse set directly on Tanimoto distance; the count is a
*budget*, and coverage is judged by nearest-representative distance, not silhouette.

INCREMENTAL: the existing subset is passed as `firstPicks`, so MaxMin only ADDS the
next N most-diverse molecules — the already-computed descriptors/ΔG are never wasted.

Outputs
-------
  data/library/subset_expansion.csv   the N newly picked molecules (-> label these)
  data/library/subset_v2.csv          combined existing + new subset
  figs/coverage_expansion.png         nearest-rep Tanimoto-distance coverage, before vs after

Usage
-----
  python pipeline/expand_subset.py --n-add 200
  python pipeline/expand_subset.py --n-add 150 --existing data/library/subset.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

RDLogger.DisableLog("rdApp.*")
# Neutral carbonyl C and O + carbon neighbour: a genuine R-CHO. Rejects carbonyl
# oxides / ketene-like charged-carbonyl species (`C=[O+][O-]`, `C=C=O`) that the
# loose `=O` SMARTS otherwise admits and that MaxMin over-selects as "diverse".
CHO = Chem.MolFromSmarts("[CX3H1;+0](=[O;+0])[#6]")
REPO = Path(__file__).resolve().parent.parent


def fingerprints(smiles: list[str], radius: int, n_bits: int):
    """ExplicitBitVect fingerprints for valid single-CHO aldehydes + kept indices.

    Excludes isotope-labeled molecules (e.g. [14C], [2H], [17O]): they are
    electronically identical to their unlabeled forms (redundant for ΔG /
    descriptors) yet read as 'diverse' in fingerprint space, so MaxMin
    over-selects them (~7x), and they can break the benzoin-SMILES builder.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or len(mol.GetSubstructMatches(CHO)) != 1:
            continue
        if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
            continue
        fps.append(gen.GetFingerprint(mol))
        keep.append(i)
    return fps, keep


def nearest_rep_distance(pool_fps, rep_positions, sample_idx):
    """For each sampled molecule, 1 - max Tanimoto to any representative."""
    reps = [pool_fps[i] for i in rep_positions]
    out = np.empty(len(sample_idx))
    for j, si in enumerate(sample_idx):
        sims = DataStructs.BulkTanimotoSimilarity(pool_fps[si], reps)
        out[j] = 1.0 - max(sims)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=str(REPO / "data/library/aldehydes_clean.csv"))
    ap.add_argument("--existing", default=str(REPO / "data/library/subset.csv"))
    ap.add_argument("--n-add", type=int, default=200, help="new molecules to add")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--cov-sample", type=int, default=20000, help="library sample for coverage")
    ap.add_argument("--expansion-out", default=str(REPO / "data/library/subset_expansion.csv"))
    ap.add_argument("--combined-out", default=str(REPO / "data/library/subset_v2.csv"))
    ap.add_argument("--fig-out", default=str(REPO / "figs/coverage_expansion.png"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    lib = pd.read_csv(args.library)
    smi_col = "SMILES" if "SMILES" in lib.columns else "smiles"
    print(f"Library: {len(lib):,} rows from {Path(args.library).name}")

    fps, keep = fingerprints(lib[smi_col].tolist(), args.radius, args.n_bits)
    pool = lib.iloc[keep].reset_index(drop=True)          # aligned with fps
    pos_of_index = {idx: p for p, idx in enumerate(pool["index"].tolist())}
    print(f"Pool (valid single-CHO): {len(pool):,}  fps ready  [{time.time()-t0:.0f}s]")

    existing = pd.read_csv(args.existing)
    first = [pos_of_index[i] for i in existing["index"] if i in pos_of_index]
    n_have = len(first)
    print(f"Existing subset: {len(existing)}  mapped to pool: {n_have}")
    if n_have != len(existing):
        print(f"  WARNING: {len(existing)-n_have} existing reps not found in pool (skipped as seeds)")

    total = n_have + args.n_add
    print(f"MaxMin: seed {n_have} (firstPicks) -> pick {total} total (+{args.n_add} new)...")
    picker = rdSimDivPickers.MaxMinPicker()
    picks = list(picker.LazyBitVectorPick(fps, len(fps), total,
                                          firstPicks=first, seed=args.seed))
    new_pos = picks[n_have:]
    new = pool.iloc[new_pos].copy()
    print(f"Picked {len(new)} new diverse molecules  [{time.time()-t0:.0f}s]")

    # ── Coverage: nearest-representative Tanimoto distance, before vs after ──
    rng = np.random.default_rng(args.seed)
    n_s = min(args.cov_sample, len(pool))
    samp = rng.choice(len(pool), n_s, replace=False)
    d_before = nearest_rep_distance(fps, first, samp)
    d_after = nearest_rep_distance(fps, first + new_pos, samp)
    def pct(d): return f"mean={d.mean():.3f} median={np.median(d):.3f} p95={np.percentile(d,95):.3f} max={d.max():.3f}"
    print(f"\nCoverage (1 - Tanimoto to nearest rep, n={n_s} sample):")
    print(f"  before ({n_have} reps): {pct(d_before)}")
    print(f"  after  ({n_have+len(new)} reps): {pct(d_after)}")
    print(f"  p95 nearest-rep distance {d_before.mean():.3f} -> improvement "
          f"{np.percentile(d_before,95)-np.percentile(d_after,95):+.3f}")

    out_cols = [c for c in ["index", smi_col, "PubChem_CID"] if c in new.columns]
    Path(args.expansion_out).parent.mkdir(parents=True, exist_ok=True)
    new[out_cols].to_csv(args.expansion_out, index=False)
    print(f"\nWrote {args.expansion_out}  ({len(new)} new molecules to label)")

    combined = pd.concat([existing[[c for c in out_cols if c in existing.columns]],
                          new[out_cols]], ignore_index=True).drop_duplicates("index")
    combined.to_csv(args.combined_out, index=False)
    print(f"Wrote {args.combined_out}  ({len(combined)} total reps)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        bins = np.linspace(0, max(d_before.max(), d_after.max()), 40)
        ax.hist(d_before, bins=bins, alpha=0.5, label=f"before ({n_have} reps)")
        ax.hist(d_after, bins=bins, alpha=0.5, label=f"after ({n_have+len(new)} reps)")
        ax.set_xlabel("nearest-representative distance (1 - Tanimoto)")
        ax.set_ylabel("# library molecules (sample)")
        ax.set_title("Chemical-space coverage: MaxMin expansion")
        ax.legend()
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.fig_out, dpi=130)
        print(f"Wrote {args.fig_out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"Done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
