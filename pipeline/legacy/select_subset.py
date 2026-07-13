#!/usr/bin/env python3
"""
A — Fingerprint-based representative subset selection.

Pipeline (SMILES only — no xTB/DFT needed, runs in minutes on the full library):

  1. Read the master aldehyde CSV.
  2. Keep valid aldehydes with exactly one -CHO group ([CX3H1]=O).
  3. Morgan/ECFP fingerprints (radius 2, fpSize bits).
  4. PCA -> low-dimensional embedding (keeps subsequent clustering cheap/robust).
  5. MiniBatchKMeans with an adaptive number of clusters k (silhouette on a
     subsample over a candidate grid, unless --k is given).
  6. One representative per cluster = the real molecule closest to the cluster
     centroid in PCA space (medoid). These span chemical space and become the
     training set for the expensive descriptor + ΔG calculations.

Outputs
-------
  ml/data/subset.csv          index,SMILES,PubChem_CID,cluster   (one row per representative)
  ml/data/embedding.parquet   index,SMILES,PC1..PCk,cluster      (all kept molecules; for plots/reuse)
  ml/figs/pca_clusters.png    PC1-PC2 scatter with representatives highlighted

Usage
-----
  python ml/select_subset.py                                   # full library, auto-k
  python ml/select_subset.py --max 5000                        # quick dry-run
  python ml/select_subset.py --k 400                           # fixed cluster count
  python ml/select_subset.py --input aldehyde/aldehydes_benzoin.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

RDLogger.DisableLog("rdApp.*")

CHO = Chem.MolFromSmarts("[CX3H1](=[O])")


def compute_fingerprints(smiles: list[str], n_bits: int, radius: int):
    """Return (fp_matrix uint8 [M,n_bits], keep_idx) for valid single-CHO aldehydes."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        if len(mol.GetSubstructMatches(CHO)) != 1:
            continue
        fps.append(gen.GetFingerprintAsNumPy(mol))
        keep.append(i)
    if not fps:
        return np.empty((0, n_bits), dtype=np.uint8), []
    return np.vstack(fps).astype(np.uint8), keep


def choose_k(emb: np.ndarray, grid: list[int], seed: int) -> int:
    """Pick k maximizing silhouette on a subsample over a candidate grid."""
    n = len(emb)
    rng = np.random.default_rng(seed)
    sample = emb if n <= 10000 else emb[rng.choice(n, 10000, replace=False)]
    best_k, best_s = grid[0], -1.0
    print("  k-scan (silhouette on subsample):")
    for k in grid:
        if k >= len(sample):
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=3)
        labels = km.fit_predict(sample)
        s = silhouette_score(sample, labels, sample_size=min(5000, len(sample)),
                             random_state=seed)
        print(f"    k={k:4d}  silhouette={s:.4f}")
        if s > best_s:
            best_k, best_s = k, s
    print(f"  -> chosen k={best_k} (silhouette={best_s:.4f})")
    return best_k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="aldehyde/aldehydes_benzoin.csv")
    ap.add_argument("--subset-out", default="ml/data/subset.csv")
    ap.add_argument("--embedding-out", default="ml/data/embedding.parquet")
    ap.add_argument("--fig-out", default="ml/figs/pca_clusters.png")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--k", type=int, default=0, help="Fixed cluster count; 0 = auto")
    ap.add_argument("--k-grid", default="200,300,400,500,600,800")
    ap.add_argument("--max", type=int, default=0, help="Cap input rows (dry-run)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.input)
    if args.max:
        df = df.head(args.max)
    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
    print(f"Loaded {len(df):,} rows from {args.input}")

    fps, keep = compute_fingerprints(df[smiles_col].tolist(), args.n_bits, args.radius)
    kept = df.iloc[keep].reset_index(drop=True)
    print(f"Valid single-CHO aldehydes: {len(kept):,}  "
          f"(fingerprints {fps.shape})  [{time.time()-t0:.0f}s]")
    if len(kept) < 10:
        print("Too few molecules to cluster.")
        return 1

    n_comp = min(args.pca_components, fps.shape[1], len(kept) - 1)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=args.seed)
    emb = pca.fit_transform(fps.astype(np.float32))
    var = pca.explained_variance_ratio_.sum()
    print(f"PCA -> {n_comp} comps, explained variance {var:.3f}  [{time.time()-t0:.0f}s]")

    grid = [int(x) for x in args.k_grid.split(",")]
    k = args.k if args.k > 0 else choose_k(emb, grid, args.seed)
    k = min(k, len(kept))

    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=2048, n_init=5)
    labels = km.fit_predict(emb)

    # Representative = molecule closest to each cluster centroid (medoid in PCA space).
    rep_local, _ = pairwise_distances_argmin_min(km.cluster_centers_, emb)
    reps = kept.iloc[rep_local].copy()
    reps["cluster"] = np.arange(k)
    print(f"Clusters: {k}   representatives: {len(reps)}  [{time.time()-t0:.0f}s]")

    out_cols = [c for c in ["index", smiles_col, "PubChem_CID"] if c in reps.columns] + ["cluster"]
    Path(args.subset_out).parent.mkdir(parents=True, exist_ok=True)
    reps[out_cols].to_csv(args.subset_out, index=False)
    print(f"Wrote {args.subset_out}  ({len(reps)} representatives)")

    # Full embedding (PC1..PCk + cluster) for plots / later reuse.
    emb_df = kept[[c for c in ["index", smiles_col, "PubChem_CID"] if c in kept.columns]].copy()
    for j in range(min(n_comp, 10)):
        emb_df[f"PC{j+1}"] = emb[:, j]
    emb_df["cluster"] = labels
    Path(args.embedding_out).parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(args.embedding_out, index=False)
    print(f"Wrote {args.embedding_out}")

    # Diagnostic scatter.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(emb[:, 0], emb[:, 1], s=3, c=labels, cmap="tab20", alpha=0.3, linewidths=0)
        ax.scatter(emb[rep_local, 0], emb[rep_local, 1], s=40, c="black",
                   marker="x", label="representatives")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"ECFP PCA — {len(kept):,} aldehydes, k={k}")
        ax.legend(loc="best")
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.fig_out, dpi=130)
        print(f"Wrote {args.fig_out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"Done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
