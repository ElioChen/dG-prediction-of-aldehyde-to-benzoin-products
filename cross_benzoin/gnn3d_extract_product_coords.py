#!/usr/bin/env python
"""
Phase 2 of the DGT-inspired 3D-GNN exploration (see gnn3d_extract_product_geometry.py's
docstring for the full backstory and memory gnn3d-dgt-inspired-ablation-20260721.md for
why Phase 1 -- real bond LENGTHS injected as an extra GINEConv edge feature -- gave only
a directionally-promising, statistically-unconfirmed signal, P=0.7976).

User diagnosis (2026-07-21, confirmed correct): Phase 1's weakness is twofold --
(1) bond-length-only captures nothing about NON-bonded atom pairs (two substituents that
are topologically far apart but spatially close after folding), which is exactly the kind
of information 2D topology can never see and 3D geometry uniquely provides; (2) GINEConv
is local (1-hop) message passing, so even if more 3D info were added to edge features, the
architecture has no mechanism to relate atoms beyond a few bonds apart. DGT's actual 43.9%
MAE-reduction result came from a full pairwise atom-atom distance bias inside a GLOBAL
self-attention mechanism, not from bond-feature augmentation.

This script extracts full heavy-atom 3D COORDINATES (not just bonded distances) for
product molecules, reusing gnn3d_extract_product_geometry.py's exact validated
atom-index-correspondence logic (heavy-atom order in the xyz file == RDKit's
Chem.MolFromSmiles(smi).GetAtoms() order, verified per-row via element-symbol match).
Downstream (gnn_architectures.py's distance-biased self-attention layer) computes the
FULL N x N pairwise distance matrix from these coordinates at train time -- storing raw
coordinates here (not the O(N^2) matrix) keeps the cache small and format-agnostic.

Usage
  python cross_benzoin/gnn3d_extract_product_coords.py \
      --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
      --rounds round2 round3 round4 round8 round9 \
      --out data/cross_benzoin/gnn3d/product_coords_r2348_r9.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent

from gnn3d_extract_product_geometry import _read_xyz, build_xyz_index  # noqa: E402


def _heavy_atom_coords(smiles: str, xyz_path: str):
    """Returns (n_heavy, flat list of x,y,z,x,y,z,...) or None if unreadable/unvalidated.
    Same validation as gnn3d_extract_product_geometry.py's _bond_lengths()."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None or m.GetNumAtoms() == 0:
        return None
    atoms = _read_xyz(xyz_path)
    if atoms is None or len(atoms) < m.GetNumAtoms():
        return None
    n_heavy = m.GetNumAtoms()
    for i in range(n_heavy):
        if m.GetAtomWithIdx(i).GetSymbol().upper() != atoms[i][0].upper():
            return None
    coords = []
    for _, x, y, z in atoms[:n_heavy]:
        coords += [x, y, z]
    return n_heavy, coords


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.table, columns=["id", "smiles", "round"])
    df = df[df["round"].isin(args.rounds)].copy()
    print(f"table rows in requested rounds: {len(df)}")

    xyz_index: dict[str, str] = {}
    for rtag in args.rounds:
        rd = REPO / "data/cross_benzoin" / f"cross_{rtag}"
        idx = build_xyz_index(rd)
        print(f"{rtag}: indexed {len(idx)} xyz_prod files under {rd}")
        xyz_index.update(idx)

    n_matched = n_valid = 0
    out_rows = []
    for rid, smi in zip(df["id"], df["smiles"]):
        xp = xyz_index.get(rid)
        if xp is None:
            continue
        n_matched += 1
        res = _heavy_atom_coords(smi, xp)
        if res is None:
            continue
        n_valid += 1
        n_heavy, coords = res
        out_rows.append({"id": rid, "n_heavy": n_heavy, "coords_flat": coords})

    print(f"matched: {n_matched}/{len(df)} ({n_matched/max(len(df),1):.1%})")
    print(f"validated: {n_valid}/{n_matched} ({n_valid/max(n_matched,1):.1%} of matched)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.out)
    print(f"wrote {len(out_df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
