#!/usr/bin/env python3
"""
Merge round9's per-chunk products.csv into one cross_round9_products.csv.

Round9 needed a two-pass featurize: the original 534-chunk array (job 24772091,
unthrottled) hit /scratch-local per-node quota exhaustion on 3,069/16,000 rows
(see cb-featurize-scratch-local-quota-20260720 memory); a retry2 array (job
24779895, 103 chunks, %60 throttle) recomputed exactly those 3,069 rows cleanly.

This script merges BOTH sources, dropping the original's rows for whichever
pairs were retried and splicing in retry2's fresh output for those same keys.
The key set for "which pairs were retried" comes from
cross_round9_retry_pairs.csv (the actual retry INPUT file) rather than from
detecting error rows in the original output: cb_featurize.py's outer exception
handler (`except Exception as exc: row = {"id": ..., "error": ...}`) writes a
bare row with NO donor_smiles/acceptor_smiles at all for a mid-conformer-search
crash (like the disk-quota exception), so those columns read back as NaN and
can't be used to identify which pair failed -- only `retry_pairs.csv` (a clean
INPUT file, no NaNs) reliably says which pairs were retried.

Usage:
    python cross_benzoin/merge_round9_products.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RDIR = REPO / "data/cross_benzoin/cross_round9"


def _err(df: pd.DataFrame) -> pd.Series:
    return df["error"].astype("string").fillna("") if "error" in df.columns else pd.Series([""] * len(df))


def main() -> int:
    orig_fs = sorted(glob.glob(str(RDIR / "chunk_*/products.csv")))
    if not orig_fs:
        print("ERROR: no original chunk_*/products.csv found", file=sys.stderr)
        return 1
    orig = pd.concat([pd.read_csv(f, low_memory=False) for f in orig_fs], ignore_index=True)
    print(f"original: merged {len(orig_fs)} chunks -> {len(orig)} rows "
          f"({(_err(orig) != '').sum()} with error)")

    retry_fs = sorted(glob.glob(str(RDIR / "retry2/chunk_*/products.csv")))
    if not retry_fs:
        print("ERROR: no retry2/chunk_*/products.csv found", file=sys.stderr)
        return 1
    retry = pd.concat([pd.read_csv(f, low_memory=False) for f in retry_fs], ignore_index=True)
    print(f"retry2: merged {len(retry_fs)} chunks -> {len(retry)} rows "
          f"({(_err(retry) != '').sum()} with error)")

    retry_pairs_csv = RDIR / "cross_round9_retry_pairs.csv"
    rp = pd.read_csv(retry_pairs_csv)
    key = ["donor_smiles", "acceptor_smiles"]
    retried_keys = set(map(tuple, rp[key].values))
    print(f"retry_pairs.csv: {len(rp)} pairs designated for retry "
          f"({len(retried_keys)} unique keys)")

    retry_keys_actual = set(map(tuple, retry[key].dropna().values))
    if retried_keys != retry_keys_actual:
        print(f"WARNING: retry_pairs.csv keys ({len(retried_keys)}) != retry2 output keys "
              f"({len(retry_keys_actual)}) -- diff: {len(retried_keys ^ retry_keys_actual)}")

    # Rows whose donor/acceptor key is NaN are whole-row exception crashes (see module
    # docstring) -- they carry no usable key and must be dropped unconditionally, not just
    # when they happen to match a retried key (NaN never matches via .isin()).
    key_is_nan = orig[key].isna().any(axis=1)
    orig_key_idx = orig.set_index(key, drop=False).index
    matches_retried = orig_key_idx.isin(retried_keys)
    keep_mask = ~(key_is_nan | matches_retried)
    n_drop = (~keep_mask).sum()
    print(f"dropping {n_drop} original rows (NaN-key crashes: {key_is_nan.sum()}, "
          f"matched-a-retried-pair: {matches_retried.sum()}) -- expected total drop "
          f"{len(retried_keys)}")
    merged = pd.concat([orig[keep_mask], retry], ignore_index=True)

    n_dupe = merged.duplicated(subset=key, keep=False).sum()
    if n_dupe:
        print(f"WARNING: {n_dupe} rows still form duplicate (donor,acceptor) pairs after "
              f"merge -- dropping, keeping the LAST (retry2) occurrence")
        merged = merged.drop_duplicates(subset=key, keep="last")

    n_pairs_input = 16000  # round9 pool size, sanity check only
    if len(merged) != n_pairs_input:
        print(f"NOTE: final row count {len(merged)} != expected pool size {n_pairs_input} "
              f"(informational, not necessarily an error)")

    n_err = (_err(merged) != "").sum()
    out = RDIR / "cross_round9_products_merged.csv"
    merged.to_csv(out, index=False)
    print(f"FINAL merged: {len(merged)} rows ({n_err} with error) -> {out}")

    afs = sorted(glob.glob(str(RDIR / "chunk_*/aldehydes.csv"))) + \
        sorted(glob.glob(str(RDIR / "retry2/chunk_*/aldehydes.csv")))
    if afs:
        adf = pd.concat([pd.read_csv(f, low_memory=False) for f in afs], ignore_index=True)
        adf = adf.drop_duplicates(subset=["id"], keep="last")
        aout = RDIR / "cross_round9_aldehydes_merged.csv"
        adf.to_csv(aout, index=False)
        print(f"merged {len(afs)} chunk aldehyde files -> {len(adf)} rows -> {aout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
