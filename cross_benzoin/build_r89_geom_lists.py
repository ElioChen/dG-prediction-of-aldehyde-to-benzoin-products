#!/usr/bin/env python
"""Build the (id, xyz_path) work lists + thermal sidecars for the rounds 8-9
DFT-label recovery (see orca_sp_from_geomlist.py header).

PRODUCT leg (ready now): reads cross_round{8,9}_products.csv, keeps error-free
rows whose product geometry (unpacked from
~/benzoin_backups/cross_benzoin_xyz_archive/cross_round{8,9}_xyz_geometry.tar.gz
into data/cross_benzoin/cross_round{8,9}/chunk_*/xyz_prod/) exists, and writes
  data/cross_benzoin/cross_round{r}_recover/product_geom_list.csv   (id, xyz_path)
  data/cross_benzoin/cross_round{r}_recover/product_thermal.csv      (id, thermal_prod_Eh, donor_ik, acceptor_ik)

ALDEHYDE leg (--aldehydes, gated on Window A's 220k homo featurize): maps the
unique r8/9 donor/acceptor InChIKeys to their 0-based aldehydes_clean_v6.csv id,
locates each aldehyde geometry + xTB thermal inside the matching Window-A chunk
(data/cross_benzoin/bde_homo_product_featurize_20260902/chunk_{id//100}: geom
tar member ald_xyz/a{id%100:06d}.xyz, thermal from aldehydes.csv), and writes
  data/cross_benzoin/r89_aldehyde_recover/aldehyde_geom_list.csv  (id, xyz_path)
  data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv    (id, thermal_ald_Eh)
skipping ids whose Window-A chunk is not done yet (re-run to pick them up).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
HARTREE = 627.5094740631
WINDOW_A = REPO / "data/cross_benzoin/bde_homo_product_featurize_20260902"
CLEAN_V6 = REPO / "data/library/aldehydes_clean_v6.csv"
OLD_PREFIXES = ("/scratch-shared/schen3/benzoin-dg/",
                "/gpfs/scratch1/shared/schen3/benzoin-dg/")


def _canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def _restore_path(p: str) -> Path:
    for pre in OLD_PREFIXES:
        if p.startswith(pre):
            return REPO / p[len(pre):]
    return Path(p)


def build_products(rounds: list[int]) -> None:
    for r in rounds:
        prod = pd.read_csv(REPO / f"data/cross_benzoin/cross_round{r}/cross_round{r}_products.csv",
                           low_memory=False)
        ok = prod[prod["error"].isna()].copy()
        ok["xyz_path"] = ok["xyz_file"].map(lambda p: str(_restore_path(p)))
        # ONE directory walk of the unpacked xyz_prod tree, then in-memory membership --
        # 16k individual GPFS stat() calls take ~30 min on a hammered filesystem.
        have_set = set(map(str, (REPO / f"data/cross_benzoin/cross_round{r}").glob(
            "chunk_*/xyz_prod/*.xyz")))
        ok["exists"] = ok["xyz_path"].isin(have_set)
        have = ok[ok["exists"]].copy()
        outd = REPO / f"data/cross_benzoin/cross_round{r}_recover"
        outd.mkdir(parents=True, exist_ok=True)
        have[["id", "xyz_path"]].to_csv(outd / "product_geom_list.csv", index=False)
        th = have[["id", "donor_id", "acceptor_id"]].copy()
        th["thermal_prod_Eh"] = have["G_xtb"].astype(float) - have["xtb_energy"].astype(float)
        th = th.rename(columns={"donor_id": "donor_ik", "acceptor_id": "acceptor_ik"})
        th[["id", "thermal_prod_Eh", "donor_ik", "acceptor_ik"]].to_csv(
            outd / "product_thermal.csv", index=False)
        print(f"round{r}: {len(ok)} error-free, {len(have)} with geometry on disk "
              f"({len(ok) - len(have)} missing) -> {outd}/product_geom_list.csv")


def _ik_to_libid() -> dict[str, int]:
    cv = pd.read_csv(CLEAN_V6, usecols=["InChIKey"])
    return {ik: i for i, ik in enumerate(cv["InChIKey"].astype(str))}


def build_aldehydes(rounds: list[int]) -> None:
    iks: set[str] = set()
    for r in rounds:
        prod = pd.read_csv(REPO / f"data/cross_benzoin/cross_round{r}/cross_round{r}_products.csv",
                           low_memory=False)
        ok = prod[prod["error"].isna()]
        iks |= set(ok["donor_id"].dropna().astype(str))
        iks |= set(ok["acceptor_id"].dropna().astype(str))
    print(f"unique r{'+'.join(map(str, rounds))} aldehyde InChIKeys: {len(iks)}")

    ik2id = _ik_to_libid()
    ids = sorted({ik2id[ik] for ik in iks if ik in ik2id})
    print(f"  mapped to clean_v6 ids: {len(ids)} ({len(iks) - len(ids)} unmapped)")

    outd = REPO / "data/cross_benzoin/r89_aldehyde_recover"
    (outd / "ald_xyz").mkdir(parents=True, exist_ok=True)
    geom_rows, therm_rows, pending = [], [], 0
    by_chunk: dict[int, list[int]] = {}
    for i in ids:
        by_chunk.setdefault(i // 100, []).append(i)

    for ch, chunk_ids in sorted(by_chunk.items()):
        cdir = WINDOW_A / f"chunk_{ch:04d}"
        tar = cdir / "geom.tar.zst"
        acsv = cdir / "aldehydes.csv"
        if not (tar.exists() and acsv.exists()):
            pending += len(chunk_ids)
            continue
        adf = pd.read_csv(acsv)
        # aldehydes.csv row order == pairs.csv row order == within-chunk geom index
        adf = adf.reset_index(drop=True)
        idcol = "index" if "index" in adf.columns else adf.columns[0]
        for i in chunk_ids:
            hit = adf.index[adf[idcol].astype("Int64") == i].tolist()
            if not hit:
                pending += 1
                continue
            row = hit[0]
            member = f"ald_xyz/a{row:06d}.xyz"
            dst = outd / "ald_xyz" / f"a{i:08d}.xyz"
            if not dst.exists():
                rc = subprocess.run(["tar", "--zstd", "-xf", str(tar), "-O", member],
                                    capture_output=True)
                if rc.returncode != 0 or not rc.stdout:
                    pending += 1
                    continue
                dst.write_bytes(rc.stdout)
            g = adf.at[row, "G_ald_xtb"] if "G_ald_xtb" in adf.columns else adf.at[row, "G_xtb"]
            e = adf.at[row, "xtb_energy"]
            if pd.isna(g) or pd.isna(e):
                pending += 1
                continue
            geom_rows.append({"id": i, "xyz_path": str(dst)})
            therm_rows.append({"id": i, "thermal_ald_Eh": float(g) - float(e)})

    pd.DataFrame(geom_rows).to_csv(outd / "aldehyde_geom_list.csv", index=False)
    pd.DataFrame(therm_rows).to_csv(outd / "aldehyde_thermal.csv", index=False)
    print(f"  ready now: {len(geom_rows)} aldehyde geoms extracted; {pending} still pending "
          f"a Window-A chunk. -> {outd}/aldehyde_geom_list.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 9])
    ap.add_argument("--aldehydes", action="store_true",
                    help="also build the aldehyde leg from Window A chunks (partial ok)")
    args = ap.parse_args()
    build_products(args.rounds)
    if args.aldehydes:
        build_aldehydes(args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
