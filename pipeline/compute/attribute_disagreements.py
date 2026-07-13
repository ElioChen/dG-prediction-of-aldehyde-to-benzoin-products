#!/usr/bin/env python3
"""
Attribute the funnel_v2-vs-CREST dG_orca disagreements: for each flagged molecule,
rebuild the benzoin and re-run BOTH conformer searches, then check whether each
method's LOWEST conformer (the Boltzmann-label driver) has the correct topology.

The method whose lowest conformer is BROKEN (bond formed/broken or fragmented) is the
one whose label is poisoned. A spuriously-low broken conformer drives the species G
DOWN, so for the benzoin PRODUCT that makes ΔG = G_bz − G_ald too NEGATIVE — i.e. the
method giving the more-negative ΔG on a disagreement is the suspect.

Driven by the indices in runs/conf_method_compare.csv (|Δ| > --thresh). funnel_v2 is
deterministic; CREST is re-run, its MTD is reproducible enough for attribution.

Usage
  python pipeline/compute/attribute_disagreements.py --thresh 5.0
"""
from __future__ import annotations
import argparse, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import thermo_orca as Th
import conf_funnel_v2 as v2
import conf_crest as cr
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

XTB = "/home/schen3/xtb/bin/xtb"
SOLV = "dmso"
TMP = "/scratch-shared/schen3/tmp_diag"
REPO = Path(__file__).resolve().parent.parent.parent


def topo(mol):
    rw = Chem.RWMol(mol)
    for a in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(a)
    m = rw.GetMol()
    return (m.GetNumBonds(), len(Chem.GetMolFrags(m)),
            tuple(sorted(a.GetDegree() for a in m.GetAtoms())))


def ref_topo(smi):
    return topo(Chem.AddHs(Chem.MolFromSmiles(smi)))


def xyz_topo(xyz):
    try:
        m = Chem.MolFromXYZBlock(xyz)
        if m is None:
            return None
        rdDetermineBonds.DetermineConnectivity(m, charge=0)
        return topo(m)
    except Exception:
        return None


def assess(tag, confs, ref):
    if not confs:
        return f"{tag}: NO CONFORMERS"
    rows = [(E, xyz_topo(x) != ref) for x, E in confs]
    nbroken = sum(b for _, b in rows)
    loE, lo_broken = rows[0]
    intactE = next((E for E, b in rows if not b), None)
    extra = ""
    if lo_broken and intactE is not None:
        extra = f"  (broken min {(intactE-loE)*627.509:+.1f} kcal below lowest intact)"
    return (f"{tag}: n={len(confs):2d} broken={nbroken:2d} "
            f"LOWEST_BROKEN={'YES' if lo_broken else 'no':3s} loE={loE:.5f}{extra}")


def run_one(idx, ald, d_fun, d_crest):
    bz = Th._make_benzoin_smiles(ald)
    ref = ref_topo(bz)
    print(f"\n=== idx {idx}  Δ={d_crest-d_fun:+.2f} (fun={d_fun:.2f} crest={d_crest:.2f})")
    print(f"    ald  {ald}")
    print(f"    bz   {bz}  ref bonds={ref[0]} frags={ref[1]}")
    with tempfile.TemporaryDirectory(dir=TMP) as wd:
        wd = Path(wd)
        cf = v2.rank_conformers_funnel_v2(bz, wd / "v2", XTB, solvent=SOLV, l10=20)
        print("  " + assess("funnel_v2", cf, ref))
        cc = cr.rank_conformers_crest(bz, wd / "crest", XTB, solvent=SOLV,
                                      cores=8, method="gfn2//gfnff", l10=20)
        print("  " + assess("crest     ", cc, ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "runs/conf_method_compare.csv"))
    ap.add_argument("--thresh", type=float, default=5.0)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    fun = "dG_orca_kcal_fun" if "dG_orca_kcal_fun" in df else "dG_xtb_kcal_fun"
    cre = "dG_orca_kcal_crest" if "dG_orca_kcal_crest" in df else "dG_xtb_kcal_crest"
    flagged = df[(df[cre] - df[fun]).abs() > a.thresh].copy()
    flagged["absd"] = (flagged[cre] - flagged[fun]).abs()
    flagged = flagged.sort_values("absd", ascending=False)
    print(f"Attributing {len(flagged)} molecules with |Δ| > {a.thresh} kcal")
    Path(TMP).mkdir(parents=True, exist_ok=True)
    for _, r in flagged.iterrows():
        try:
            run_one(int(r["index"]), r["SMILES"], r[fun], r[cre])
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
