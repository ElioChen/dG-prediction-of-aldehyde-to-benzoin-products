#!/usr/bin/env python
"""
Extract REAL 3D bond lengths for cross-benzoin PRODUCT molecules from the xTB-funnel
-optimised geometries already computed by cb_featurize.py (xyz_prod/*.xyz per chunk),
as a per-bond feature cache keyed by row id (donor_id__acceptor_id).

Motivated by the DGT paper (Zhang & Lapkin, Nat Commun 2026, s41467-026-75005-9): the
SINGLE largest lever they found for molecular property prediction was augmenting a graph
transformer with precise (DFT-derived) 3D structural information -- bond lengths, atom-
atom distances, bond-bond angles -- yielding a 43.9% average MAE reduction on QM9 HOMO/
LUMO, dwarfing the dual-graph-representation (3-6%) and RPE/RSE (3-7.5%) contributions.
This project's GNN (train_cross_gnn.py/TripleGNN, gnn_architectures.py/TripleGNNAttn,
the current production champion) builds graphs from bare SMILES via RDKit and has NEVER
used the 3D geometry this pipeline already pays to compute for every product molecule
(xTB-funnel-optimised, GFN2 level -- not full DFT, but real conformer geometry, well
beyond the MMFF/UFF baseline DGT itself showed still gives ~23% of the DFT-level gain).
This is a natural, low-risk augmentation to try in GPU-idle windows between DFT-SP
campaigns (see docs/HANDOFF_round10_20260721_*.md; GPU idle while round10's CPU-bound
featurize/DFT arrays run on genoa).

Scope decision: PRODUCT side only, not donor/acceptor. Donor/acceptor xyz geometries
were only ever persisted for the small ~0.15-0.3% of aldehydes that were cache MISSES in
a given round (see cb-featurize-scratch-local-quota memory) -- the other ~99.7% were
served from homo_v6's aldehydes_all.csv cache, whose original per-molecule xyz files were
cleaned up long ago (ORCA/xTB scratch hygiene) and are not recoverable without re-running
~220k xTB geometry optimisations. Product xyz, by contrast, is ALWAYS freshly computed
(products are unique per pair, never cached) and persisted per round's chunk_*/xyz_prod/,
so full coverage exists for any round whose scratch wasn't manually cleaned afterward.

Coverage note (checked 2026-07-21): rounds 2/3/4/8/9(+10 in flight) still have their
xyz_prod/ directories on disk; rounds 1/5/6/7 do not (cleaned at some point, cause not
investigated -- out of scope for this side experiment). This script only processes rounds
with coverage; the resulting GNN experiment is trained/evaluated on that matched subset,
compared against the current champion architecture RESTRICTED to the same subset (honest
ablation, not a different-N comparison).

Atom-index correspondence: conf_funnel_v2's embedding step does `Chem.AddHs(Chem.
MolFromSmiles(smiles))` then embeds -- RDKit's AddHs APPENDS new H atoms after all
existing (heavy) atom indices, never reordering them. So heavy-atom index i in the xyz
file's atom block corresponds directly to `Chem.MolFromSmiles(smiles).GetAtoms()[i]`,
the SAME heavy-atom-only mol used by train_cross_gnn.graph() to build the 2D topology
graph. Every row is validated by comparing element symbols position-by-position before
its bond lengths are trusted; any mismatch (parser version drift, sanitisation
differences) invalidates that row rather than silently injecting wrong distances.

Usage
  python cross_benzoin/gnn3d_extract_product_geometry.py \
      --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
      --rounds round2 round3 round4 round8 round9 \
      --out data/cross_benzoin/gnn3d/product_bond_lengths_r2349_r9.parquet
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent


def _read_xyz(path: str) -> list[tuple[str, float, float, float]] | None:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 3:
        return None
    try:
        n = int(lines[0].strip())
    except ValueError:
        return None
    atoms = []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        if len(parts) < 4:
            return None
        sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((sym, x, y, z))
    return atoms if len(atoms) == n else None


def _bond_lengths(smiles: str, xyz_path: str) -> list[float] | None:
    """Returns bond lengths in the SAME order as Chem.MolFromSmiles(smiles).GetBonds()
    (matching train_cross_gnn.graph()'s bond-feature iteration order), or None if the
    xyz is missing/unreadable or the heavy-atom element sequence doesn't validate."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None or m.GetNumAtoms() == 0:
        return None
    atoms = _read_xyz(xyz_path)
    if atoms is None or len(atoms) < m.GetNumAtoms():
        return None
    n_heavy = m.GetNumAtoms()
    for i in range(n_heavy):
        rd_sym = m.GetAtomWithIdx(i).GetSymbol()
        xyz_sym = atoms[i][0]
        if rd_sym.upper() != xyz_sym.upper():
            return None  # atom-index correspondence broken -- don't trust this row
    coords = [(x, y, z) for _, x, y, z in atoms[:n_heavy]]
    lens = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        xi, yi, zi = coords[i]
        xj, yj, zj = coords[j]
        lens.append(((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) ** 0.5)
    return lens


def build_xyz_index(round_dir: Path) -> dict[str, str]:
    """id (donor_id__acceptor_id) -> absolute xyz path, globbing chunk_*/xyz_prod and any
    retry*/chunk_*/xyz_prod subdirectories (retries win -- glob order + dict overwrite is
    fine since retries are the LAST-run, most-trustworthy copy, same convention as
    merge_round9_products.py's "keep last on duplicate" rule)."""
    idx = {}
    patterns = [
        str(round_dir / "chunk_*" / "xyz_prod" / "prod_*.xyz"),
        str(round_dir / "retry*" / "chunk_*" / "xyz_prod" / "prod_*.xyz"),
    ]
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            stem = Path(p).stem  # "prod_<donor_id>__<acceptor_id>"
            rid = stem[len("prod_"):]
            idx[rid] = p
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--rounds", nargs="+", required=True,
                     help="round tags matching the table's 'round' column, e.g. round2 round9")
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

    n_matched = 0
    n_valid = 0
    out_rows = []
    for rid, smi in zip(df["id"], df["smiles"]):
        xp = xyz_index.get(rid)
        if xp is None:
            continue
        n_matched += 1
        lens = _bond_lengths(smi, xp)
        if lens is None:
            continue
        n_valid += 1
        out_rows.append({"id": rid, "n_bonds": len(lens), "bond_lengths": lens})

    print(f"matched to an xyz file: {n_matched}/{len(df)} "
          f"({n_matched / max(len(df), 1):.1%})")
    print(f"passed element-symbol validation: {n_valid}/{n_matched} "
          f"({n_valid / max(n_matched, 1):.1%} of matched)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.out)
    print(f"wrote {len(out_df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
