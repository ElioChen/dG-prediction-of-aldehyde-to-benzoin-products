#!/usr/bin/env python3
"""
Merge backfilled Multiwfn columns into a chunked screen's features.csv files.
=============================================================================
For each chunk_*/ in a screen dir, fill the (empty) adch_*/qtaim_* columns of
features.csv from the matching ald_multiwfn.csv, joining on `index`.

Usage
-----
  python merge_multiwfn.py --screen-dir /abs/screen_v6            # -> features.csv (in place)
  python merge_multiwfn.py --screen-dir /abs/screen_v6 --suffix _mwf  # -> features_mwf.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import ald_descriptors_qm as A

MWF_COLS = list(A._MWF_FIELDS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--suffix", default="",
                    help="output suffix (default '': overwrite features.csv)")
    args = ap.parse_args()

    sd = Path(args.screen_dir)
    chunks = sorted(sd.glob("chunk_*"))
    n_done = n_skip = n_filled = 0
    for ch in chunks:
        feat, mwf = ch / "features.csv", ch / "ald_multiwfn.csv"
        if not feat.exists() or not mwf.exists():
            n_skip += 1
            continue
        df = pd.read_csv(feat, dtype={"index": str})
        m = pd.read_csv(mwf, dtype={"index": str})
        cols = [c for c in MWF_COLS if c in m.columns]
        m = m[["index"] + cols].set_index("index")
        df = df.set_index("index")
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
            df.loc[m.index.intersection(df.index), c] = m[c]
        df = df.reset_index()
        out = feat if not args.suffix else ch / f"features{args.suffix}.csv"
        df.to_csv(out, index=False)
        n_done += 1
        n_filled += int(df["adch_CHO_C"].notna().sum()) if "adch_CHO_C" in df else 0
    print(f"merged {n_done} chunks ({n_skip} skipped, missing inputs); "
          f"{n_filled} rows with ADCH total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
