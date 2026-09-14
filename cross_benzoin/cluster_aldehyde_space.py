#!/usr/bin/env python
"""Similarity-based clustering of the full 220,859-aldehyde library + a
regression diagnostic against the homo dG target, requested 2026-09-14 to
serve the flying-dataset diversity/split question (CHEMICAL_SPACE.md sec8):
does the chemotype clustering used to stratify the flying-dataset splits
actually carry signal for the reaction property, or is it orthogonal to it?

Homo reactions are self-couplings (donor_smiles == acceptor_smiles in
homo_relabel_pairs.csv), so `homo_sp_labels.csv`'s `id` is a single
`ald_idx` -- the dG_r2scan_kcal target is legitimately a per-aldehyde
quantity here, unlike the cross project's pair-level target. That is what
makes "cluster aldehydes, regress dG" a well-posed question for homo and not
for cross.

Pipeline
  1. Morgan (ECFP4, r=2, 2048 bit) fingerprints for all 220,859 aldehydes
     (smiles_canonical, RDKit, multiprocess).
  2. MiniBatchKMeans on the bit vectors -- approximate/scalable clustering,
     O(N*k) not O(N^2) Tanimoto-pairwise (chosen scale: full library, per
     user 2026-09-14). This is an approximation: Euclidean k-means on binary
     ECFP bits is a standard scalable proxy for Tanimoto clustering, not
     equivalent to it -- flagged, not silently treated as exact.
  3. Join cluster_id -> dG_r2scan_kcal via the current homo_sp_labels.csv
     (partial: campaign in flight, re-run this script as coverage grows).
  4. Diagnostic regressions, 5-fold CV, same folds throughout for a fair
     comparison:
       a. global-mean baseline
       b. cluster-mean-only (fold-safe: cluster means computed on the train
          fold, not leaked from test)
       c. XGBoost on [cluster_id one-hot] -- can trees exploit cluster
          non-linearly beyond the per-cluster mean?
       d. XGBoost on [fingerprint folded to 128 bits] -- full structural
          signal, no cluster binning, as an upper-reference for "how much
          does raw structure explain" (NOT a champion-quality model: no QM
          descriptors, no baseline subtraction, 5-fold not scaffold-disjoint;
          it's a bracket, not a candidate for deployment).

Outputs (data/chemical_space/):
  aldehyde_fingerprints_ecfp4_2048.npy   (220859, 2048) uint8, ald_idx-ordered
  aldehyde_clusters_k{K}.parquet         ald_idx, cluster_id
  homo_cluster_dG_regression.md          the diagnostic report

Usage
    PY=/home/schen3/venv/nhc-workflow/bin/python
    $PY cross_benzoin/cluster_aldehyde_space.py --k 150
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
CS = REPO / "data" / "chemical_space"
INDEX = CS / "aldehyde_index.parquet"
LABELS = REPO / "data" / "cross_benzoin" / "homo_standalone" / "relabel_sp" / "homo_sp_labels.csv"
FP_BITS = 2048
FP_RADIUS = 2


def _fp_worker(args: tuple[int, str]) -> tuple[int, np.ndarray | None]:
    idx, smi = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return idx, arr


def compute_fingerprints(df: pd.DataFrame, n_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fp_matrix[N,2048] uint8, ok_mask[N] bool). Failed SMILES get an
    all-zero row and ok_mask=False -- excluded from clustering, kept in the
    index so downstream joins by position stay aligned."""
    n = len(df)
    fps = np.zeros((n, FP_BITS), dtype=np.uint8)
    ok = np.zeros((n,), dtype=bool)
    jobs = list(enumerate(df["smiles_canonical"].tolist()))
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for i, (idx, arr) in enumerate(pool.imap_unordered(_fp_worker, jobs, chunksize=200)):
            if arr is not None:
                fps[idx] = arr
                ok[idx] = True
            if (i + 1) % 20000 == 0:
                print(f"  fp {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"fingerprints: {ok.sum()}/{n} parsed ok ({time.time()-t0:.0f}s)", flush=True)
    return fps, ok


def cluster(fps: np.ndarray, ok: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    cluster_id = np.full((fps.shape[0],), -1, dtype=np.int32)
    X = fps[ok].astype(np.float32)
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5, max_iter=200)
    labels = km.fit_predict(X)
    cluster_id[ok] = labels
    print(f"MiniBatchKMeans k={k}: inertia={km.inertia_:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return cluster_id


def fold_fingerprints(fps: np.ndarray, factor: int = 16) -> np.ndarray:
    """Fold an (N, bits) binary fingerprint array down to (N, bits/factor) by
    OR-ing consecutive chunks -- the standard cheminformatics fingerprint-
    folding trick, done here in place of a TruncatedSVD/PCA reduction.
    Deterministic and data-independent (unlike PCA, nothing is *fit*), so it
    needs no per-fold refit and cannot leak test-fold structure. Chosen over
    PCA 2026-09-14 after PCA turned out to need BLAS matmuls this venv's numpy
    executes unvectorized on this node's CPU generation (a plain 76000x2048
    matmul benchmarked at 14s instead of the <1s it should take) -- folding
    is pure numpy reshape+reduce, no BLAS, ~0.5s for the full labeled set."""
    n, bits = fps.shape
    assert bits % factor == 0
    return fps.reshape(n, bits // factor, factor).max(axis=2)


def _one_row(y, tr, te, cluster_id, fp_folded, k, seed):
    from sklearn.metrics import mean_absolute_error, r2_score
    import xgboost as xgb

    y_tr, y_te = y[tr], y[te]
    row = {}

    # (a) global mean
    pred = np.full_like(y_te, y_tr.mean())
    row["global_mean"] = (mean_absolute_error(y_te, pred), r2_score(y_te, pred))

    # (b) cluster/group mean, fold-safe (train-fold means only; unseen group -> global mean)
    means = pd.Series(y_tr).groupby(cluster_id[tr]).mean()
    global_mean = y_tr.mean()
    pred = np.array([means.get(c, global_mean) for c in cluster_id[te]])
    row["group_mean"] = (mean_absolute_error(y_te, pred), r2_score(y_te, pred))

    # (c) XGB on group one-hot
    cats = range(k) if k else sorted(pd.unique(cluster_id))
    oh_tr = pd.get_dummies(pd.Categorical(cluster_id[tr], categories=cats)).to_numpy(dtype=np.float32)
    oh_te = pd.get_dummies(pd.Categorical(cluster_id[te], categories=cats)).to_numpy(dtype=np.float32)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed, n_jobs=8)
    m.fit(oh_tr, y_tr)
    pred = m.predict(oh_te)
    row["xgb_group_onehot"] = (mean_absolute_error(y_te, pred), r2_score(y_te, pred))

    # (d) XGB on folded ECFP-128 (structure bracket, no group binning)
    m2 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=8)
    m2.fit(fp_folded[tr].astype(np.float32), y_tr)
    pred = m2.predict(fp_folded[te].astype(np.float32))
    row["xgb_fp_folded128"] = (mean_absolute_error(y_te, pred), r2_score(y_te, pred))

    return row


def cv_diagnostics(merged: pd.DataFrame, fps: np.ndarray, k: int, seed: int) -> dict:
    """Runs the 4-method bracket under three split regimes:
      random    -- plain 5-fold KFold (optimistic: same cluster/scaffold can
                   appear in both train and test)
      cluster   -- GroupKFold on cluster_id (structurally novel test chemotypes)
      scaffold  -- GroupKFold on the existing Bemis-Murcko scaffold_split_legacy
                   (a second, independently-computed notion of "similar
                   structure", free -- already in aldehyde_index.parquet)
    Comparing cluster vs scaffold GroupKFold cross-validates the clustering
    *method* itself (2026-09-14 ask): if they tell the same story, the choice
    of grouping method doesn't matter much for this target; if they diverge,
    it does and needs a closer look (e.g. Butina) before trusting either.
    """
    from sklearn.model_selection import KFold, GroupKFold

    y = merged["dG_r2scan_kcal"].to_numpy()
    cluster_id = merged["cluster_id"].to_numpy()
    scaffold_id = merged["scaffold_code"].to_numpy()
    fp_rows = fps[merged["_fp_row"].to_numpy()]
    fp_folded = fold_fingerprints(fp_rows, factor=16)  # (n, 128), computed once -- see fold_fingerprints docstring
    print(f"  folded fingerprints {fp_rows.shape} -> {fp_folded.shape}", flush=True)

    regimes = {
        "random": (KFold(n_splits=5, shuffle=True, random_state=seed).split(y), cluster_id, k),
        "cluster_disjoint": (GroupKFold(n_splits=5).split(y, groups=cluster_id), cluster_id, k),
        "scaffold_disjoint": (GroupKFold(n_splits=5).split(y, groups=scaffold_id), scaffold_id, 0),
    }

    summary = {}
    for regime_name, (splitter, group_arr, k_arg) in regimes.items():
        results = {"global_mean": [], "group_mean": [], "xgb_group_onehot": [], "xgb_fp_folded128": []}
        for tr, te in splitter:
            row = _one_row(y, tr, te, group_arr, fp_folded, k_arg, seed)
            for name, val in row.items():
                results[name].append(val)
        summary[regime_name] = {}
        for name, vals in results.items():
            maes = [v[0] for v in vals]
            r2s = [v[1] for v in vals]
            summary[regime_name][name] = {
                "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
                "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
            }
        print(f"  regime={regime_name} done", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fp-cache", default=str(CS / f"aldehyde_fingerprints_ecfp4_{FP_BITS}.npy"))
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N aldehydes only (0=full)")
    ap.add_argument("--reuse-clusters", action="store_true",
                     help="skip re-running MiniBatchKMeans; load the existing aldehyde_clusters_k{K}.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(INDEX).sort_values("ald_idx").reset_index(drop=True)
    assert (df["ald_idx"].to_numpy() == np.arange(len(df))).all(), "ald_idx must be 0..N-1 positional"
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
        assert (df["ald_idx"].to_numpy() == np.arange(len(df))).all()
    print(f"index: {len(df)} aldehydes", flush=True)

    fp_cache = Path(args.fp_cache)
    if args.limit and fp_cache == Path(str(CS / f"aldehyde_fingerprints_ecfp4_{FP_BITS}.npy")):
        fp_cache = CS / f"_smoke_fp_limit{args.limit}.npy"  # never collide with the full-library cache
    if fp_cache.exists():
        print(f"loading cached fingerprints {fp_cache}", flush=True)
        fps = np.load(fp_cache)
        ok = fps.any(axis=1)
    else:
        fps, ok = compute_fingerprints(df, args.workers)
        np.save(fp_cache, fps)
        print(f"wrote {fp_cache}", flush=True)

    clu_out = CS / f"aldehyde_clusters_k{args.k}.parquet"
    if args.reuse_clusters and clu_out.exists():
        clu_df = pd.read_parquet(clu_out)
        print(f"reusing cached clusters {clu_out}", flush=True)
    else:
        cluster_id = cluster(fps, ok, args.k, args.seed)
        clu_df = pd.DataFrame({"ald_idx": df["ald_idx"], "cluster_id": cluster_id})
        clu_df.to_parquet(clu_out)
        print(f"wrote {clu_out}", flush=True)

    sizes = clu_df.loc[clu_df.cluster_id >= 0, "cluster_id"].value_counts()
    print(f"cluster sizes: n={args.k} min={sizes.min()} median={sizes.median():.0f} "
          f"max={sizes.max()} (unparsed excluded: {(~ok).sum()})", flush=True)

    # join labels (id == ald_idx for homo self-reactions)
    labels = pd.read_csv(LABELS, dtype={"id": str})
    n_before = len(labels)
    labels["id"] = pd.to_numeric(labels["id"], errors="coerce")
    n_bad = labels["id"].isna().sum()
    if n_bad:
        print(f"WARNING: {n_bad}/{n_before} rows in {LABELS.name} have a non-numeric id "
              f"(e.g. literal 'nan') -- dropped, not a bug in this script", flush=True)
    labels = labels.dropna(subset=["id"])
    labels["id"] = labels["id"].astype("int64")
    labels = labels.rename(columns={"id": "ald_idx"})
    merged = labels.merge(clu_df, on="ald_idx", how="inner")
    merged = merged.merge(df[["ald_idx", "scaffold"]], on="ald_idx", how="left")
    merged = merged[merged["cluster_id"] >= 0].reset_index(drop=True)
    n_no_scaffold = merged["scaffold"].isna().sum()
    merged = merged.dropna(subset=["scaffold"]).reset_index(drop=True)
    merged["scaffold_code"] = pd.factorize(merged["scaffold"])[0]
    merged["_fp_row"] = merged["ald_idx"].to_numpy()  # fps is ald_idx-positional
    coverage = len(merged) / len(df)
    print(f"labels joined: {len(merged)}/{len(labels)} labeled rows matched a valid cluster "
          f"({coverage:.1%} of the full 220,859-aldehyde library); {n_no_scaffold} dropped for "
          f"missing Bemis-Murcko scaffold; {merged['scaffold_code'].nunique()} unique scaffolds "
          f"in the labeled subset", flush=True)

    diag = cv_diagnostics(merged, fps, args.k, args.seed)
    print(json.dumps(diag, indent=2), flush=True)
    diag_json = CS / "homo_cluster_dG_regression_diag.json"
    diag_json.write_text(json.dumps(
        {"diag": diag, "n_labeled": len(merged), "n_total": len(df), "k": args.k,
         "coverage": len(merged) / len(df)}, indent=2))
    print(f"wrote {diag_json}", flush=True)

    report = CS / "homo_cluster_dG_regression.md"
    regime_labels = {
        "random": "random 5-fold (optimistic -- same cluster/scaffold can leak across train/test)",
        "cluster_disjoint": "GroupKFold on KMeans cluster_id (structurally novel test chemotypes)",
        "scaffold_disjoint": "GroupKFold on Bemis-Murcko scaffold (independent grouping method, free -- "
                              "already computed in aldehyde_index.parquet)",
    }
    method_labels = {
        "global_mean": "global mean (no structure)",
        "group_mean": "group-mean only (cluster or scaffold, per regime)",
        "xgb_group_onehot": "XGB on group one-hot",
        "xgb_fp_folded128": "XGB on fingerprint folded to 128 bits (structure bracket)",
    }
    rows = []
    for regime, methods in diag.items():
        for m, v in methods.items():
            rows.append(f"| {regime} | {method_labels[m]} | {v['MAE_mean']:.3f} +/- {v['MAE_std']:.3f} | {v['R2_mean']:.3f} |")
    table = "\n".join(rows)

    with open(report, "w") as f:
        f.write(f"""# Homo aldehyde clustering vs dG_r2scan_kcal -- diagnostic

Generated {pd.Timestamp.now(tz='UTC').isoformat()}. Requested 2026-09-14: does
chemotype clustering (for the flying-dataset split design, CHEMICAL_SPACE.md
sec8) carry signal for the homo reaction target, or is it orthogonal -- and
does the choice of clustering *method* (KMeans-on-ECFP4 vs the existing
Bemis-Murcko scaffold) and split regime (random vs group-disjoint) change
the answer?

**Caveat: partial labels.** The homo SP relabel campaign is in flight.
{len(merged)}/{len(df)} aldehydes ({coverage:.1%}) have a dG_r2scan_kcal
label right now -- re-run this script once the campaign drains for the
full-library numbers; treat these as directional, not final.

## Method
- {ok.sum()}/{len(df)} aldehydes fingerprinted (ECFP4 r=2, {FP_BITS} bit),
  {(~ok).sum()} SMILES failed to parse and were excluded from clustering.
- MiniBatchKMeans, k={args.k} (Euclidean on binary bits -- an approximation
  of Tanimoto clustering, not equivalent to it; chosen for O(N*k) scalability
  at full 220,859-molecule scale per 2026-09-14 direction). Cluster sizes:
  min {int(sizes.min())}, median {sizes.median():.0f}, max {int(sizes.max())}.
- Second grouping method: Bemis-Murcko scaffold (`aldehyde_index.parquet`
  `scaffold` column, already computed for this library by
  `build_scaffold_splits.py`/BDE work) -- {merged['scaffold_code'].nunique()}
  unique scaffolds in the labeled subset, {n_no_scaffold} rows dropped for a
  missing scaffold.
- y=dG_r2scan_kcal (homo self-reaction free energy; homo pairs are
  donor==acceptor so `id` in homo_sp_labels.csv is a single ald_idx -- a
  legitimately per-aldehyde target here, unlike cross's pair-level target).
- Three split regimes x four methods, 5 folds each, same regime-specific
  folds reused across all four methods for a fair within-regime comparison.
  The structure bracket uses ECFP4 folded 2048->128 bits (OR-fold of
  consecutive chunks) rather than a PCA/SVD reduction -- folding is
  data-independent so it needs no per-fold refit and cannot leak test-fold
  structure; switched from an earlier fold-safe-PCA version of this script
  after PCA turned out to need BLAS matmuls this venv's numpy executes
  unvectorized on this node's CPU generation (14s for a single 76000x2048
  matmul that should take <1s), making even one PCA refit minutes long.

## Results (MAE in kcal/mol)

| split regime | method | MAE | R2 |
|---|---|---|---|
{table}

(`xgb_group_onehot` / `xgb_fp_folded128` are diagnostic brackets, not
champion-quality models -- no QM descriptors, no baseline subtraction.
Compare their *gap* within a regime, not their absolute MAE, to the champion
pipeline.)

## Reading

**Method-validation (this session's ask):** compare the `cluster_disjoint`
row block to the `scaffold_disjoint` block. If they tell a similar story
(similar MAE/R2 gaps from `random`), the KMeans-on-ECFP4 clustering and the
independently-computed Bemis-Murcko scaffold agree on how hard structural
extrapolation is for this target -- the clustering *method* choice doesn't
matter much here, and there's no need to spend the ~20-40 min for an exact
Butina/Tanimoto clustering just to validate it further. If they diverge, the
method choice does matter and Butina is the next step.

**Split-difficulty:** compare `random` to the two `*_disjoint` regimes. A
much worse `group_mean`/`xgb_group_onehot` MAE under `*_disjoint` (like the
scaffold-disjoint correction found for BDE/cross,
[[bde-scaffold-leakage-finding]], [[cross-r1-10-champion]]) means a
structurally-grouped flying-dataset split is the honest one to build, and a
random split would overstate how well any 2D-structure-only model
generalizes. A small gap instead means 2D structure just doesn't carry much
signal for `dG_r2scan_kcal` at all (compare all four `xgb_fp_folded128` R2
values to the near-zero `global_mean` R2) -- consistent with
[[homo_active_relabel_null_result]]'s redirect: the signal this target needs
is in QM/electronic descriptors the champion pipeline computes, not in gross
2D chemotype similarity.
""")
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
