#!/usr/bin/env python
"""Compute extra morfeus 3D descriptors (Pyramidalization + atom-resolved Dispersion
P_int) for a homo_v6 table, reusing the geometry already saved (xyz_file) — no
re-optimization. Written as a non-destructive sidecar keyed by `id`.

The existing cb_featurize morfeus tier (vbur/sterimol/SASA/whole-molecule P_int) is
already computed for all products/aldehydes. This adds the two NEW orthogonal
descriptors validated on a 15-molecule pilot: SolidAngle/ConeAngle were tested and
fail systematically (0/15) at ketC/carbC — those two are ligand-cone-angle metrics
for metal-coordination sites and don't apply to a fully-substituted organic carbon
(the central atom's own substituents violate the vdW-non-overlap assumption they
rely on) — so they are intentionally excluded here.

Usage:
  python add_morfeus_descriptors.py products   # -> products_morfeus_extra.csv
  python add_morfeus_descriptors.py aldehydes  # -> aldehydes_morfeus_extra.csv
"""
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from morfeus import Pyramidalization, Dispersion

_COMPUTE = Path(__file__).resolve().parents[2] / "pipeline" / "compute"
sys.path.insert(0, str(_COMPUTE))
import ald_descriptors_qm as A         # noqa: E402
import featurize_product as FP         # noqa: E402

ROOT = Path("data/cross_benzoin/homo_v6")


def _atoms_product(symbols, coords):
    core = FP.find_benzoin_core(symbols, coords)
    if core is None:
        return None
    return {"ketC": core["ketC"], "carbC": core["carbC"]}


def _atoms_aldehyde(symbols, coords):
    hits = A.find_aldehyde_atoms(symbols, coords)
    if not hits:
        return None
    return {"CHO_C": hits[0][0]}


def _row(args):
    xyz_file, which = args
    fields = ["pyr", "pyr_angle", "disp_p_int"]
    names = ["ketC", "carbC"] if which == "products" else ["CHO_C"]
    out = {f"{f}_{n}": np.nan for f in fields for n in names}
    if not xyz_file or not os.path.exists(xyz_file):
        return out
    try:
        xyz = Path(xyz_file).read_text(encoding="utf-8")
        symbols, coords = A.parse_xyz(xyz)
        atoms = _atoms_product(symbols, coords) if which == "products" else _atoms_aldehyde(symbols, coords)
        if atoms is None:
            return out
        disp = Dispersion(symbols, coords)
        for name, idx0 in atoms.items():
            idx1 = idx0 + 1  # morfeus is 1-based
            try:
                pyr = Pyramidalization(coords, idx1, elements=symbols)
                out[f"pyr_{name}"] = pyr.P
                out[f"pyr_angle_{name}"] = pyr.P_angle
            except Exception:
                pass
            try:
                out[f"disp_p_int_{name}"] = disp.atom_p_int.get(idx1, np.nan)
            except Exception:
                pass
    except Exception:
        pass
    return out


def main(which):
    if which == "products":
        src, out = ROOT / "products_all.csv", ROOT / "products_morfeus_extra.csv"
    elif which == "aldehydes":
        src, out = ROOT / "aldehydes_all.csv", ROOT / "aldehydes_morfeus_extra.csv"
    else:
        sys.exit("arg must be 'products' or 'aldehydes'")

    df = pd.read_csv(src, usecols=["id", "xyz_file", "error"], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df.drop_duplicates("id").reset_index(drop=True)
    print(f"{which}: {len(df):,} rows, computing morfeus extras", flush=True)

    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("NPROC") or 16)
    with Pool(nproc) as pool:
        rows = pool.map(_row, [(xf, which) for xf in df["xyz_file"]], chunksize=64)

    desc = pd.DataFrame(rows)
    result = pd.concat([df[["id"]].reset_index(drop=True), desc], axis=1)
    result.to_csv(out, index=False)
    key_col = "pyr_ketC" if which == "products" else "pyr_CHO_C"
    ok = int(result[key_col].notna().sum())
    print(f"wrote {out}  ({len(result):,} rows x {desc.shape[1]} cols; "
          f"{ok:,} filled, {len(result)-ok:,} missing)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "products")
