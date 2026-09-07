#!/usr/bin/env python3
"""Geometry coverage for the full 35,528-row champion training table -- decides the
full B97-3c recompute strategy (archived-SP-only vs regen vs hybrid)."""
import glob, os, subprocess, tarfile
from pathlib import Path
import pandas as pd
from rdkit import Chem

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
TAB = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
ARCH = Path("/home/schen3/benzoin_backups/cross_benzoin_xyz_archive")
HOME_REC = Path("/home/schen3/benzoin_backups/recovery_20260902")

df = pd.read_parquet(TAB, columns=["id", "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles", "round", "new_scaffold_split"])
print(f"{len(df)} pairs, splits: {df['new_scaffold_split'].value_counts().to_dict()}")

# ---- product geom index: on-disk + round tarballs ----
prod = set()
for p in glob.glob(str(REPO / "data/cross_benzoin/cross_round*/**/xyz_prod/prod_*.xyz"), recursive=True):
    prod.add(os.path.basename(p)[5:-4])
n_disk = len(prod)
for tb in sorted(ARCH.glob("cross_round*_xyz_geometry.tar.gz")):
    out = subprocess.run(["tar", "tzf", str(tb)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "/xyz_prod/prod_" in line and line.endswith(".xyz"):
            prod.add(os.path.basename(line)[5:-4])
print(f"product geoms indexed: {n_disk} on-disk + tar = {len(prod)} total")

def has_prod(pid):
    a, b = pid.split("__") if "__" in pid else (pid, pid)
    return pid in prod or f"{b}__{a}" in prod

df["prod_geom"] = df["id"].map(has_prod)

# ---- aldehyde geom index: recover dirs (a<8d>.xyz) + round tarball xyz_ald + home recover tars ----
ald_geom_ids = set()   # numeric ids as int
for d in [REPO / "data/cross_benzoin/r89_aldehyde_recover/ald_xyz",
          REPO / "data/cross_benzoin/r10_aldehyde_recover/ald_xyz"]:
    for f in d.glob("a*.xyz"):
        try: ald_geom_ids.add(int(f.stem[1:]))
        except ValueError: pass
ald_geom_names = set()  # by InChIKey stem from round tarball xyz_ald/ald_<key>.xyz
for tb in sorted(ARCH.glob("cross_round*_xyz_geometry.tar.gz")):
    out = subprocess.run(["tar", "tzf", str(tb)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "/xyz_ald/ald_" in line and line.endswith(".xyz"):
            ald_geom_names.add(os.path.basename(line)[4:-4])
for tb in HOME_REC.glob("*aldehyde*geometry*.tar.gz"):
    try:
        with tarfile.open(tb) as t:
            for n in t.getnames():
                if n.endswith(".xyz") and "/xyz_ald/" in n:
                    ald_geom_names.add(os.path.basename(n)[4:-4])
    except Exception as e:
        print("  (skip", tb.name, e, ")")
print(f"aldehyde geoms: {len(ald_geom_ids)} by-numeric-id (recover dirs) + {len(ald_geom_names)} by-name (tarball xyz_ald)")

# aldehyde id map: canonical smiles -> numeric id (aldehydes_all.csv)
ald_by_smi = {}
import csv as _csv
for r in _csv.DictReader(open(REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv", encoding="utf-8-sig")):
    m = Chem.MolFromSmiles(r.get("smiles") or "")
    if m: ald_by_smi[Chem.MolToSmiles(m)] = r.get("id")

def ald_ok(smi, ikey_hint):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return False
    c = Chem.MolToSmiles(m)
    if ikey_hint and ikey_hint in ald_geom_names: return True
    nid = ald_by_smi.get(c)
    if nid not in (None, ""):
        try:
            return int(float(nid)) in ald_geom_ids
        except ValueError:
            return False
    return False

df["don_geom"] = [ald_ok(s, k) for s, k in zip(df["donor_smiles"], df["donor_id"])]
df["acc_geom"] = [ald_ok(s, k) for s, k in zip(df["acceptor_smiles"], df["acceptor_id"])]
df["all3"] = df["prod_geom"] & df["don_geom"] & df["acc_geom"]

print("\n=== coverage ===")
print(f"  product geom      : {df['prod_geom'].mean():.1%}  ({df['prod_geom'].sum()})")
print(f"  donor ald geom    : {df['don_geom'].mean():.1%}")
print(f"  acceptor ald geom : {df['acc_geom'].mean():.1%}")
print(f"  ALL 3 geoms       : {df['all3'].mean():.1%}  ({df['all3'].sum()})")
print("\n=== all3 by split ===")
print(df.groupby("new_scaffold_split")["all3"].agg(["mean", "sum", "count"]))
print("\n=== all3 by round ===")
print(df.groupby("round")["all3"].agg(["mean", "sum", "count"]))
