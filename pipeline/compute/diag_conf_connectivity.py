#!/usr/bin/env python3
"""Diagnostic: does the v2 funnel's denser GFN-FF sampling produce conformers with
BROKEN connectivity (isomerized/fragmented), and does the spuriously-low-energy
broken conformer become the 'global minimum' that poisons the Boltzmann label?

For a few production-relevant molecules where v2-K10 disagreed badly with the v1
high-K consensus, rebuild the benzoin product, run the v2 funnel, and for every
returned conformer perceive its bond graph (rdDetermineBonds) and compare to the
input SMILES graph. Report: #broken, and whether the LOWEST conformer is broken.
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import thermo_orca as Th
import conf_funnel_v2 as v2

XTB = "/home/schen3/xtb/bin/xtb"
SOLV = "dmso"

# Worst production-relevant (non-SF5) v2-K10 vs v1-consensus disagreements:
ALD_SMILES = [
    "C1=CC=C2C=C3C(=CC2=C1)C(=O)C(=CO3)C=O",          # fused, v1cons +5.6 -> v2 +1.3
    "CC(=O)OC1=C(NC(=C1)C2=CC=CC=C2)C=O",             # acetoxy pyrrole, +11.8 -> +8.3
]

def topo(mol):
    """Bond-ORDER-INDEPENDENT topology fingerprint of the heavy-atom graph:
    (n_heavy_bonds, n_fragments, sorted heavy-atom degree sequence). Robust to
    aromaticity/bond-order perception; only changes if a bond actually forms/breaks
    or the molecule fragments."""
    rw = Chem.RWMol(mol)
    # drop H atoms (highest index first) so degrees count heavy neighbours only
    for a in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1], reverse=True):
        rw.RemoveAtom(a)
    m = rw.GetMol()
    deg = tuple(sorted(a.GetDegree() for a in m.GetAtoms()))
    nbonds = m.GetNumBonds()
    nfrag = len(Chem.GetMolFrags(m))
    return (nbonds, nfrag, deg)

def ref_topo(smiles):
    m = Chem.MolFromSmiles(smiles)
    return topo(Chem.AddHs(m))

def perceive_topo(xyz_str, charge=0):
    try:
        m = Chem.MolFromXYZBlock(xyz_str)
        if m is None: return None
        rdDetermineBonds.DetermineConnectivity(m, charge=charge)
        return topo(m)
    except Exception as e:
        return f"ERR:{e}"

def run_one(ald):
    bz = Th._make_benzoin_smiles(ald)
    print(f"\n=== aldehyde: {ald}\n    benzoin : {bz}")
    if not bz: print("  (no benzoin)"); return
    ref = ref_topo(bz)
    print(f"    ref topo (n_bonds,n_frag,degseq-len): ({ref[0]},{ref[1]},{len(ref[2])})")
    with tempfile.TemporaryDirectory(dir="/scratch-shared/schen3/tmp_diag") as wd:
        confs = v2.rank_conformers_funnel_v2(bz, Path(wd), XTB, solvent=SOLV, l10=20)
    if not confs:
        print("  (no conformers returned)"); return
    rows = []
    for xyz, E in confs:
        t = perceive_topo(xyz)
        broken = (t != ref)        # topology changed: bond formed/broken or fragmented
        rows.append((E, broken, t))
    nbroken = sum(1 for _, b, _ in rows if b)
    lo_E, lo_broken, lo_t = rows[0]
    print(f"  conformers kept: {len(rows)} | topology CHANGED: {nbroken}")
    print(f"  LOWEST conformer (the label driver): E={lo_E:.6f}  topo_changed={lo_broken}")
    if lo_broken:
        print(f"    ref topo: {ref}")
        print(f"    got topo: {lo_t}  (n_frag>1 => fragmented; bond count differs => iso)")
    intact = [(E, t) for E, b, t in rows if not b]
    if intact and lo_broken:
        gap = (intact[0][0] - lo_E) * 627.509
        print(f"    lowest INTACT E={intact[0][0]:.6f}  -> changed conf is {gap:+.1f} kcal LOWER (spurious)")

if __name__ == "__main__":
    Path("/scratch-shared/schen3/tmp_diag").mkdir(parents=True, exist_ok=True)
    for a in ALD_SMILES:
        try: run_one(a)
        except Exception as e:
            import traceback; traceback.print_exc()
