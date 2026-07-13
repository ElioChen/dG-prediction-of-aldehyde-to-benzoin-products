#!/usr/bin/env python
"""Build a retry manifest of DFT-SP molecules that FAILED/timed out in the full run.

Job 24178884 ran with a 3600s ORCA soft-timeout (below the 7200s safe floor), so
~4% of molecules were written to chunk_*.csv with error='orca_sp_failed', dG=NaN.
The chunk resume guard never retries them, so collect them here and resubmit with
TIMEOUT=7200 into a SEPARATE results dir (preserve-output-history: never overwrite).

Run AFTER the main array is fully drained:
    python pipeline/compute/build_retry_manifest.py
Then:
    N=$(python -c "import pandas as pd;print(len(pd.read_parquet(
        '/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/manifest_retry.parquet')))")
    CH=96; NT=$(( (N+CH-1)/CH ))
    sbatch --array=0-$((NT-1))%128 pipeline/slurm/submit_dft_sp_retry.sh
"""
import glob
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3")
MANIFEST = RESULTS / "manifest.parquet"
OUT = RESULTS / "manifest_retry.parquet"


def main() -> int:
    chunks = sorted(glob.glob(str(RESULTS / "chunk_*.csv")))
    if not chunks:
        print("no chunk_*.csv found", file=sys.stderr)
        return 1

    failed_ids, scored, present = set(), 0, 0
    for f in chunks:
        d = pd.read_csv(f, usecols=["id", "dG_orca_kcal"])
        scored += len(d)
        bad = d[d["dG_orca_kcal"].isna()]
        present += int((~d["dG_orca_kcal"].isna()).sum())
        failed_ids.update(bad["id"].tolist())

    man = pd.read_parquet(MANIFEST)
    n_total = len(man)
    # CRITICAL: manifest id is stored as object/str while chunk-CSV id is int64;
    # coerce both to a common int key or isin() silently matches nothing (-> whole library flagged).
    man["_idk"] = pd.to_numeric(man["id"], errors="coerce").astype("Int64")
    failed_ids = {int(x) for x in failed_ids if pd.notna(x)}
    retry = man[man["_idk"].isin(failed_ids)].reset_index(drop=True)

    # molecules never reached at all (missing chunk CSV) -> also retry
    seen_ids = set()
    for f in chunks:
        seen_ids.update(int(x) for x in pd.read_csv(f, usecols=["id"])["id"].tolist())
    never = man[~man["_idk"].isin(seen_ids)]
    if len(never):
        retry = pd.concat([retry, never], ignore_index=True).drop_duplicates("id")

    retry = retry.drop(columns=["_idk"]).reset_index(drop=True)
    retry.to_parquet(OUT, index=False)
    n_chunks = (len(retry) + 95) // 96
    print(f"manifest total       : {n_total}")
    print(f"chunk CSVs scanned   : {len(chunks)}  (rows={scored}, dG ok={present})")
    print(f"failed/NaN molecules : {len(failed_ids)}")
    print(f"never-scored (gap)   : {len(never)}")
    print(f"retry manifest       : {len(retry)} mols -> {OUT}")
    print(f"suggested array      : 0-{n_chunks - 1}%128  (CH=96)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
