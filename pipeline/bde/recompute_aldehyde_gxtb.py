#!/usr/bin/env python3
"""Cheap G_gxtb recompute for aldehydes the 2026-09-02 BDE featurize rebuild left
without a whole-molecule g-xTB free energy (see memory
predict-dg-g-gxtb-regression-fixed / RUN_LOG_20260903.md 09-07 entry).

That rebuild's featurize_aldehyde() runs a GFN2 --ohess (conformer opt + Hessian,
the EXPENSIVE step) and saves G_ald_xtb (the GFN2 free energy) plus the optimized
geometry (archived in chunk_*/geom.tar.zst:ald_xyz/aNNNNNN.xyz) -- but discards the
raw ohess stdout, so the GFN2 *electronic* energy at that geometry (E_el) was never
saved, and no g-xTB energy was ever computed at all.

This script does NOT re-run the expensive ohess. It reuses the already-optimized
geometry and this project's established hybrid-correction recipe
(pipeline/compute/gxtb_baseline.py::_species): two cheap single-points (no opt, no
Hessian) at the SAME fixed geometry --

  1. GFN2 --sp --alpb dmso            -> E_el   (reproduces the ohess run's own
                                                  electronic energy at that geometry,
                                                  deterministic)
  2. g-xTB --gxtb --sp --cosmo dmso   -> E_gxtb

  G_gxtb = E_gxtb + (G_ald_xtb - E_el)   (reuse the GFN2 RRHO thermal correction --
                                           g-xTB has no native thermal pipeline here,
                                           same pattern gxtb_baseline.py established)

Usage (one chunk):
  python recompute_aldehyde_gxtb.py --chunk-dir .../chunk_0000 \
      --out .../chunk_0000/aldehydes_gxtb.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

XTB_BIN = "/home/schen3/xtb/bin/xtb"
SOLVENT = "dmso"
_E_PAT = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s*Eh\s*::", re.IGNORECASE)
_GXTB_E_PAT = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh")


def _run(cmd: list[str], cwd: Path, timeout: int) -> str:
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""
    except Exception as exc:  # pragma: no cover
        return f"EXC {exc}"


def gfn2_sp(xyz_path: Path, wd: Path, charge: int = 0, timeout: int = 300) -> float | None:
    wd.mkdir(parents=True, exist_ok=True)
    g = wd / "mol.xyz"
    g.write_text(xyz_path.read_text())
    cmd = [XTB_BIN, "mol.xyz", "--gfn", "2", "--sp", "--alpb", SOLVENT,
           "--chrg", str(charge), "--norestart"]
    out = _run(cmd, wd, timeout)
    m = _E_PAT.findall(out)
    return float(m[-1]) if m else None


def gxtb_sp(xyz_path: Path, wd: Path, charge: int = 0, timeout: int = 300) -> float | None:
    wd.mkdir(parents=True, exist_ok=True)
    g = wd / "mol.xyz"
    g.write_text(xyz_path.read_text())
    cmd = [XTB_BIN, "mol.xyz", "--gxtb", "--sp", "--cosmo", SOLVENT,
           "--chrg", str(charge), "--norestart"]
    out = _run(cmd, wd, timeout)
    m = _GXTB_E_PAT.findall(out)
    return float(m[-1]) if m else None


def charge_of(smiles: str) -> int:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        return Chem.GetFormalCharge(m) if m is not None else 0
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="cap #molecules (pilot)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    ald_csv = args.chunk_dir / "aldehydes.csv"
    if not ald_csv.exists():
        print(f"no aldehydes.csv in {args.chunk_dir}")
        return 1
    import pandas as pd
    ald = pd.read_csv(ald_csv, low_memory=False)
    ald = ald[ald.get("error", "").astype("string").fillna("") == ""]
    ald = ald[ald["G_ald_xtb"].notna()]
    if args.limit:
        ald = ald.head(args.limit)
    print(f"{len(ald)} candidate aldehydes in {args.chunk_dir.name}")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        geom_dir = args.chunk_dir / "ald_xyz"
        if not geom_dir.exists():
            tar_path = args.chunk_dir / "geom.tar.zst"
            if not tar_path.exists():
                print(f"no ald_xyz/ and no geom.tar.zst in {args.chunk_dir}")
                return 1
            subprocess.run(["tar", "-I", "zstd", "-xf", str(tar_path), "-C", str(tdp)], check=True)
            geom_dir = tdp / "ald_xyz"

        rows = []
        for _, r in ald.iterrows():
            idx = int(r["index"])
            xyz = geom_dir / f"a{idx:06d}.xyz"
            row = {"index": idx, "SMILES": r.get("SMILES"), "G_ald_xtb": r["G_ald_xtb"],
                   "E_el_gfn2": None, "E_gxtb": None, "G_gxtb": None, "note": ""}
            if not xyz.exists():
                row["note"] = "no_xyz"
                rows.append(row)
                continue
            charge = charge_of(str(r.get("SMILES", "")))
            wd = tdp / f"work_{idx:06d}"
            e_el = gfn2_sp(xyz, wd / "gfn2", charge, args.timeout)
            e_gxtb = gxtb_sp(xyz, wd / "gxtb", charge, args.timeout)
            row["E_el_gfn2"] = e_el
            row["E_gxtb"] = e_gxtb
            if e_el is not None and e_gxtb is not None:
                row["G_gxtb"] = e_gxtb + (float(r["G_ald_xtb"]) - e_el)
                row["note"] = "ok"
            else:
                row["note"] = "sp_failed"
            rows.append(row)
            # bound node-local inode use
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "SMILES", "G_ald_xtb", "E_el_gfn2",
                                          "E_gxtb", "G_gxtb", "note"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["note"] == "ok")
    print(f"wrote {len(rows)} rows ({n_ok} ok) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
