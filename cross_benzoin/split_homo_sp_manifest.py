#!/usr/bin/env python3
"""Partition homo_sp_manifest.csv into:
  homo_sp_manifest_archived.csv  -- both prod + ald xyz members are in their zst
  homo_sp_manifest_regen.csv     -- at least one member missing (geom-archive gap)

The gap set needs GFN2-opt-from-SMILES geometries (homo_sp_regen_worker.py); the
archived set goes straight to homo_sp_from_geom_worker.py.

Usage: python cross_benzoin/split_homo_sp_manifest.py
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import pandas as pd

D = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel_sp")


def members(arc: str, cache: dict) -> set:
    if arc not in cache:
        try:
            r = subprocess.run(["tar", "--zstd", "-tf", arc], capture_output=True,
                               text=True, timeout=120)
            cache[arc] = set(r.stdout.split()) if r.returncode == 0 else set()
        except Exception:
            cache[arc] = set()
    return cache[arc]


def main() -> int:
    m = pd.read_csv(D / "homo_sp_manifest.csv", dtype={"id": str})
    cache: dict = {}
    arcs = pd.unique(pd.concat([m["prod_arc"], m["ald_arc"]]))
    print(f"{len(m)} rows, {len(arcs)} unique archives to scan")
    for i, a in enumerate(arcs):
        members(a, cache)
        if (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(arcs)}", flush=True)

    ok = m.apply(lambda r: (r["prod_mem"] in members(r["prod_arc"], cache)
                            and r["ald_mem"] in members(r["ald_arc"], cache)), axis=1)
    arch, regen = m[ok].copy(), m[~ok].copy()
    arch.to_csv(D / "homo_sp_manifest_archived.csv", index=False)
    regen.to_csv(D / "homo_sp_manifest_regen.csv", index=False)
    print(f"\narchived (both members present): {len(arch)}  ({100*len(arch)/len(m):.1f}%)")
    print(f"needs regen (>=1 missing):        {len(regen)}  ({100*len(regen)/len(m):.1f}%)")
    print("regen split:", regen["new_scaffold_split"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
