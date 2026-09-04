#!/usr/bin/env python
"""Aldehyde leg of the r8/9 DFT-label recovery -- REGEN variant.

The decision (2026-09-03) was to regenerate the r8/9 aldehyde funnel_v3 xTB
geometries + thermal from scratch (SLURM 26351006) instead of harvesting them
out of Window A's 220k homo featurize (build_r89_geom_lists.py::build_aldehydes,
now stale). This reads the regen output

    data/cross_benzoin/r89_aldehyde_regen/chunk_*/aldehydes.csv

where `id` is the clean_v6 numeric lib_id, `xyz_file` is an absolute geometry
path, and `G_xtb` / `xtb_energy` are the xTB Gibbs / electronic energies (Eh),
and writes the two sidecars assemble_r89_dft_sp.py expects:

    data/cross_benzoin/r89_aldehyde_recover/aldehyde_geom_list.csv  (id, xyz_path)
    data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv    (id, thermal_ald_Eh)

thermal_ald_Eh = G_xtb - xtb_energy  (the xTB thermal/entropic correction that is
added to the r2SCAN-3c SP electronic energy downstream).

Safe to run while 26351006 is still finishing -- it just picks up fewer chunks;
re-run to refresh. Overwrites any earlier (Window-A-harvest) sidecars in place.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGEN = REPO / "data/cross_benzoin/r89_aldehyde_regen"
OUTD = REPO / "data/cross_benzoin/r89_aldehyde_recover"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-dir", default=str(REGEN))
    ap.add_argument("--out-dir", default=str(OUTD),
                    help="dir for aldehyde_geom_list.csv + aldehyde_thermal.csv")
    ap.add_argument("--check-xyz", action="store_true",
                    help="drop rows whose xyz_file is missing on disk")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    files = sorted(glob.glob(f"{args.regen_dir}/chunk_*/aldehydes.csv"))
    if not files:
        print(f"no aldehydes.csv under {args.regen_dir}/chunk_*/", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_raw = len(df)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)

    err = df.get("error")
    if err is not None:
        df = df[err.isna() | (err.astype(str).str.strip() == "")].copy()
    for c in ("G_xtb", "xtb_energy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["G_xtb"].notna() & df["xtb_energy"].notna()].copy()

    df = df.drop_duplicates("id", keep="last")

    df["xyz_path"] = df["xyz_file"].astype(str)
    if args.check_xyz:
        have = df["xyz_path"].map(lambda p: Path(p).exists())
        print(f"xyz on disk: {int(have.sum())}/{len(df)} ({int((~have).sum())} missing)")
        df = df[have].copy()

    df["thermal_ald_Eh"] = df["G_xtb"] - df["xtb_energy"]

    out_dir.mkdir(parents=True, exist_ok=True)
    df[["id", "xyz_path"]].sort_values("id").to_csv(
        out_dir / "aldehyde_geom_list.csv", index=False)
    df[["id", "thermal_ald_Eh"]].sort_values("id").to_csv(
        out_dir / "aldehyde_thermal.csv", index=False)

    print(f"{len(files)} chunks, {n_raw} raw aldehyde rows -> {len(df)} unique "
          f"error-free ids with xTB thermal")
    print(f"  thermal_ald_Eh: mean {df.thermal_ald_Eh.mean():.5f} "
          f"min {df.thermal_ald_Eh.min():.5f} max {df.thermal_ald_Eh.max():.5f}")
    print(f"  -> {out_dir}/aldehyde_geom_list.csv")
    print(f"  -> {out_dir}/aldehyde_thermal.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
