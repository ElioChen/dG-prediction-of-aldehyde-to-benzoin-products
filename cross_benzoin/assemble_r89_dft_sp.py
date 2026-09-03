#!/usr/bin/env python
"""Assemble the rounds 8-9 recovered DFT-SP labels from the three-species SP legs.

    dG_orca = (E_prod  + thermal_prod)
            - (E_donor + thermal_donor)
            - (E_acc   + thermal_acc)      [Eh]  * 627.5094740631  -> kcal/mol

Inputs (per round r in --rounds):
  data/raw/dft_sp_cross/cross_round{r}/sp_products/chunk_*.csv   id, E_orca_Eh   (product SP)
  data/cross_benzoin/cross_round{r}_recover/product_thermal.csv  id, thermal_prod_Eh, donor_ik, acceptor_ik
Shared aldehyde legs (r8 + r9 pooled):
  data/raw/dft_sp_cross/r89_aldehyde_sp/chunk_*.csv             id(libid), E_orca_Eh   (aldehyde SP)
  data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv  id(libid), thermal_ald_Eh

Output (matches assemble_cross_training_table_v3.py's round_paths() expectation):
  data/raw/dft_sp_cross/cross_round{r}/cross_round{r}_dft_sp.csv   id, dG_orca_kcal
  data/raw/dft_sp_cross/cross_round{r}/cross_round{r}_dft_sp_detail.csv   (all terms + skip reasons)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
HARTREE = 627.5094740631
CLEAN_V6 = REPO / "data/library/aldehydes_clean_v6.csv"


def _load_sp(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pat))
    if not files:
        return pd.DataFrame(columns=["id", "E_orca_Eh"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["E_orca_Eh"].notna() & (df["E_orca_Eh"].astype(str) != "")]
    df["E_orca_Eh"] = df["E_orca_Eh"].astype(float)
    return df.drop_duplicates("id", keep="last")[["id", "E_orca_Eh"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 9])
    args = ap.parse_args()

    ik2id = {ik: i for i, ik in enumerate(
        pd.read_csv(CLEAN_V6, usecols=["InChIKey"])["InChIKey"].astype(str))}

    ald_sp = _load_sp(str(REPO / "data/raw/dft_sp_cross/r89_aldehyde_sp/chunk_*.csv"))
    ald_th = pd.read_csv(REPO / "data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv")
    ald = ald_sp.merge(ald_th, on="id", how="inner")
    ald["G_ald_Eh"] = ald["E_orca_Eh"] + ald["thermal_ald_Eh"]
    g_ald = dict(zip(ald["id"].astype(int), ald["G_ald_Eh"]))
    print(f"aldehyde G table: {len(g_ald)} ids "
          f"({len(ald_sp)} SP done, {len(ald_th)} thermal known)")

    for r in args.rounds:
        prod_sp = _load_sp(str(REPO / f"data/raw/dft_sp_cross/cross_round{r}/sp_products/chunk_*.csv"))
        pth = pd.read_csv(REPO / f"data/cross_benzoin/cross_round{r}_recover/product_thermal.csv")
        m = prod_sp.merge(pth, on="id", how="inner")
        m["G_prod_Eh"] = m["E_orca_Eh"] + m["thermal_prod_Eh"]
        m["did"] = m["donor_ik"].astype(str).map(ik2id)
        m["aid"] = m["acceptor_ik"].astype(str).map(ik2id)
        m["G_donor_Eh"] = m["did"].map(g_ald)
        m["G_acc_Eh"] = m["aid"].map(g_ald)
        ok = m[m["G_donor_Eh"].notna() & m["G_acc_Eh"].notna()].copy()
        ok["dG_orca_kcal"] = (ok["G_prod_Eh"] - ok["G_donor_Eh"] - ok["G_acc_Eh"]) * HARTREE

        outd = REPO / f"data/raw/dft_sp_cross/cross_round{r}"
        outd.mkdir(parents=True, exist_ok=True)
        ok[["id", "dG_orca_kcal"]].to_csv(outd / f"cross_round{r}_dft_sp.csv", index=False)
        m.assign(has_donor_G=m["G_donor_Eh"].notna(),
                 has_acc_G=m["G_acc_Eh"].notna()).to_csv(
            outd / f"cross_round{r}_dft_sp_detail.csv", index=False)
        print(f"round{r}: {len(prod_sp)} product SP done, {len(m)} joined thermal, "
              f"{len(ok)}/{len(m)} with both aldehyde Gs -> {outd}/cross_round{r}_dft_sp.csv")
        if len(ok):
            print(f"         dG_orca_kcal: mean {ok.dG_orca_kcal.mean():.2f} "
                  f"p5 {ok.dG_orca_kcal.quantile(.05):.2f} "
                  f"p95 {ok.dG_orca_kcal.quantile(.95):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
