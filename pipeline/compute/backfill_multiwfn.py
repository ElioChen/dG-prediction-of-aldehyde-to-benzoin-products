#!/usr/bin/env python3
"""
Backfill Multiwfn (L3) descriptors onto an existing featurized set.
==================================================================
The library screen (featurize_screen.py) leaves the adch_*/qtaim_* columns EMPTY
because Multiwfn is not run there. This script computes ONLY those columns, reusing
the geometry that was already saved per molecule (the `xyz_file` column) — so there
is NO re-optimization, just xTB-molden single points + Multiwfn analysis.

Anchoring is the aldehyde CHO carbon (same find_aldehyde_atoms used by
ald_descriptors_qm.calc_multiwfn). Output: one row per molecule with `index`,
`SMILES`, and the _MWF_FIELDS, ready to merge back on `index`.

Usage
-----
  python backfill_multiwfn.py --input screen_all.csv --output mwf.csv \
      --xtb-bin /home/schen3/xtb/bin/xtb \
      --multiwfn-bin /home/schen3/mutiwfn/Multiwfn_noGUI --workers 12
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ald_descriptors_qm as A

log = logging.getLogger("backfill_multiwfn")
OUT_FIELDS = ["index", "SMILES"] + list(A._MWF_FIELDS)


def one(rec: dict, idx: int, work_dir: Path, xtb_bin: str, mwf_bin: str) -> dict:
    row = {f: None for f in OUT_FIELDS}
    row["index"] = str(rec.get("index") or rec.get("id") or idx)
    row["SMILES"] = rec.get("SMILES") or rec.get("smiles") or ""
    xyzf = (rec.get("xyz_file") or "").strip()
    if not xyzf or not os.path.exists(xyzf):
        return row
    try:
        xyz = Path(xyzf).read_text(encoding="utf-8")
        symbols, coords = A.parse_xyz(xyz)
        row.update(A.calc_multiwfn(xyz, symbols, coords, xtb_bin, mwf_bin,
                                   work_dir / f"m{idx:06d}", stem=f"m{idx:06d}"))
    except Exception as e:
        log.debug("idx=%s failed: %s", row["index"], e)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with index,SMILES,xyz_file")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--xtb-bin", default=shutil.which("xtb") or "/home/schen3/xtb/bin/xtb")
    ap.add_argument("--multiwfn-bin", default="/home/schen3/mutiwfn/Multiwfn_noGUI")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    mwf_bin = args.multiwfn_bin
    if not Path(mwf_bin).exists():
        log.error("Multiwfn not found: %s", mwf_bin); return 1

    with open(args.input, encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    if args.skip:
        records = records[args.skip:]
    if args.max:
        records = records[: args.max]
    log.info("Backfilling Multiwfn for %d molecules", len(records))

    work_dir = Path(args.work_dir or os.environ.get("TMPDIR", "/tmp")) / "backfill_mwf"
    work_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    kw = dict(work_dir=work_dir, xtb_bin=xtb_bin, mwf_bin=mwf_bin)
    n_ok = n_filled = 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()

        def emit(row):
            nonlocal n_ok, n_filled
            w.writerow(row); fh.flush()
            n_ok += 1
            n_filled += row.get("adch_CHO_C") is not None

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(one, r, i, **kw): i for i, r in enumerate(records)}
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"index": str(futs[fut])}
                    emit(row)
        else:
            for i, r in enumerate(records):
                emit(one(r, i, **kw))
                if (i + 1) % 50 == 0:
                    log.info("  %d/%d", i + 1, len(records))

    log.info("Wrote %s — %d rows, %d with ADCH", args.output, n_ok, n_filled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
