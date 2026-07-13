#!/usr/bin/env python3
"""
Backfill product-side Multiwfn (ADCH/QTAIM) descriptors onto an existing
cb_featurize products.csv, reusing the saved geometry (xyz_file) — no
conformer search / re-optimization, just xTB-molden SP + Multiwfn.

Mirrors backfill_multiwfn.py (aldehyde side) but anchors on the benzoin core
(find_benzoin_core) instead of the CHO carbon, via featurize_product's
calc_multiwfn_product.

Usage
-----
  python backfill_multiwfn_product.py --input chunk_0000/products.csv \
      --output chunk_0000/prod_multiwfn.csv \
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
import featurize_product as FP

log = logging.getLogger("backfill_multiwfn_product")
OUT_FIELDS = ["id", "smiles"] + list(FP._MWF)


def one(rec: dict, idx: int, work_dir: Path, xtb_bin: str, mwf_bin: str) -> dict:
    row = {f: None for f in OUT_FIELDS}
    row["id"] = str(rec.get("id") or idx)
    row["smiles"] = rec.get("smiles", "")
    if (rec.get("error") or "").strip():
        return row  # no valid geometry for a row that already errored
    xyzf = (rec.get("xyz_file") or "").strip()
    if not xyzf or not os.path.exists(xyzf):
        return row
    try:
        xyz = Path(xyzf).read_text(encoding="utf-8")
        symbols, coords = A.parse_xyz(xyz)
        core = FP.find_benzoin_core(symbols, coords)
        if core is None:
            return row
        stem = f"p{idx:06d}"
        row.update(FP.calc_multiwfn_product(xyz, symbols, coords, core, xtb_bin, mwf_bin,
                                            work_dir / stem, stem))
    except Exception as e:
        log.debug("id=%s failed: %s", row["id"], e)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="products.csv with id,smiles,xyz_file,error")
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
    log.info("Backfilling product Multiwfn for %d molecules", len(records))

    work_dir = Path(args.work_dir or os.environ.get("TMPDIR", "/tmp")) / "backfill_mwf_prod"
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
            n_filled += row.get("adch_ketC") is not None

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(one, r, i, **kw): i for i, r in enumerate(records)}
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except Exception:
                        row = {"id": str(futs[fut])}
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
