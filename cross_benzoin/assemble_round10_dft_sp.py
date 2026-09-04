#!/usr/bin/env python
"""Assemble round10 (fat20_stage1) recovered DFT-SP labels from the three SP legs.

    dG_orca = (E_prod + thermal_prod) - (E_donor + thermal_donor) - (E_acc + thermal_acc)
              [Eh] * 627.5094740631  -> kcal/mol

Inputs:
  data/raw/dft_sp_cross/cross_round10_fat20_stage1/sp_products/chunk_*.csv  id(IK__IK), E_orca_Eh
  data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_products_merged.csv
      id, donor_id, acceptor_id (InChIKeys), xtb_energy, G_xtb   -> thermal_prod = G_xtb - xtb_energy
  data/raw/dft_sp_cross/r10_aldehyde_sp/chunk_*.csv               id(libid), E_orca_Eh
  data/cross_benzoin/r10_aldehyde_recover/aldehyde_thermal.csv    id(libid), thermal_ald_Eh

Output:
  data/raw/dft_sp_cross/cross_round10_fat20_stage1/cross_round10_fat20_stage1_dft_sp.csv   id, dG_orca_kcal
  ..._dft_sp_detail.csv   (all terms + skip flags)
  also copies the label file to data/raw/dft_sp_cross/cross_round10/cross_round10_dft_sp.csv
  where assemble_cross_training_table_v3.py::round_paths(10) looks for it.
"""
from __future__ import annotations
import glob
import shutil
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HARTREE = 627.5094740631
CLEAN_V6 = REPO / "data/library/aldehydes_clean_v6.csv"
BASE = REPO / "data/cross_benzoin/cross_round10_fat20_stage1"
PROD_SP = REPO / "data/raw/dft_sp_cross/cross_round10_fat20_stage1/sp_products"
# round10's selection reuses aldehydes already SP'd for r8/9 -- pool both legs.
ALD_SP_DIRS = [REPO / "data/raw/dft_sp_cross/r10_aldehyde_sp",
               REPO / "data/raw/dft_sp_cross/r89_aldehyde_sp"]
ALD_TH_CSVS = [REPO / "data/cross_benzoin/r10_aldehyde_recover/aldehyde_thermal.csv",
               REPO / "data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv"]
OUTDIR = REPO / "data/raw/dft_sp_cross/cross_round10_fat20_stage1"
LEGACY_LABEL = REPO / "data/raw/dft_sp_cross/cross_round10/cross_round10_dft_sp.csv"


def _load_sp(pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(pat))
    if not files:
        return pd.DataFrame(columns=["id", "E_orca_Eh"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["E_orca_Eh"].notna() & (df["E_orca_Eh"].astype(str).str.strip() != "")]
    df["E_orca_Eh"] = df["E_orca_Eh"].astype(float)
    return df.drop_duplicates("id", keep="last")[["id", "E_orca_Eh"]]


def main() -> int:
    ik2id = {ik: i for i, ik in enumerate(
        pd.read_csv(CLEAN_V6, usecols=["InChIKey"])["InChIKey"].astype(str))}

    ald_sp = pd.concat([_load_sp(str(d / "chunk_*.csv")) for d in ALD_SP_DIRS],
                       ignore_index=True)
    ald_sp["id"] = ald_sp["id"].astype(float).astype(int)
    ald_sp = ald_sp.drop_duplicates("id", keep="last")
    ald_th = pd.concat([pd.read_csv(c) for c in ALD_TH_CSVS], ignore_index=True)
    ald_th["id"] = ald_th["id"].astype(float).astype(int)
    ald_th = ald_th.drop_duplicates("id", keep="last")
    ald = ald_sp.merge(ald_th, on="id", how="inner")
    ald["G_ald_Eh"] = ald["E_orca_Eh"] + ald["thermal_ald_Eh"]
    g_ald = dict(zip(ald["id"].astype(int), ald["G_ald_Eh"]))
    print(f"aldehyde G table: {len(g_ald)} ids "
          f"(r10+r89 pooled: {len(ald_sp)} SP, {len(ald_th)} thermal)")

    prod_sp = _load_sp(str(PROD_SP / "chunk_*.csv"))
    pm = pd.read_csv(BASE / "cross_round10_products_merged.csv",
                     usecols=["id", "donor_id", "acceptor_id", "xtb_energy", "G_xtb", "error"],
                     low_memory=False)
    pm["thermal_prod_Eh"] = pm["G_xtb"].astype(float) - pm["xtb_energy"].astype(float)
    m = prod_sp.merge(pm, on="id", how="inner")
    m["G_prod_Eh"] = m["E_orca_Eh"] + m["thermal_prod_Eh"]
    m["did"] = m["donor_id"].astype(str).map(ik2id)
    m["aid"] = m["acceptor_id"].astype(str).map(ik2id)
    m["G_donor_Eh"] = m["did"].map(g_ald)
    m["G_acc_Eh"] = m["aid"].map(g_ald)
    ok = m[m["G_prod_Eh"].notna() & m["G_donor_Eh"].notna() & m["G_acc_Eh"].notna()].copy()
    ok["dG_orca_kcal"] = (ok["G_prod_Eh"] - ok["G_donor_Eh"] - ok["G_acc_Eh"]) * HARTREE

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / "cross_round10_fat20_stage1_dft_sp.csv"
    ok[["id", "dG_orca_kcal"]].to_csv(out_csv, index=False)
    m.assign(has_prod_G=m["G_prod_Eh"].notna(),
             has_donor_G=m["G_donor_Eh"].notna(),
             has_acc_G=m["G_acc_Eh"].notna()).to_csv(
        OUTDIR / "cross_round10_fat20_stage1_dft_sp_detail.csv", index=False)
    LEGACY_LABEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_csv, LEGACY_LABEL)

    print(f"round10: {len(prod_sp)} product SP, {len(m)} joined thermal, "
          f"{len(ok)}/{len(m)} with all three Gs -> {out_csv}")
    print(f"         also copied -> {LEGACY_LABEL}")
    if len(ok):
        q = ok["dG_orca_kcal"]
        print(f"         dG_orca_kcal: mean {q.mean():.2f}  p5 {q.quantile(.05):.2f}  "
              f"p50 {q.median():.2f}  p95 {q.quantile(.95):.2f}  (n={len(q)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
