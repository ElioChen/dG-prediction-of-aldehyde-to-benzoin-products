#!/usr/bin/env python3
"""Does CREST avoid the broken-connectivity poisoning that the v2 funnel suffers?

Runs, on the same problem molecules diag_conf_connectivity.py flagged (v2-K10 vs
v1-consensus blow-ups), both the v2 RDKit funnel and CREST, and reports for each
whether the LOWEST conformer (the Boltzmann-label driver) has the correct topology.
A clean fix = CREST's lowest conformer matches the input graph where the funnel's did
not, and the two methods' lowest-conformer GFN2 energies / geometries agree."""
from __future__ import annotations
import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermo_orca as Th
import conf_funnel_v2 as v2
import conf_crest as cr
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

XTB = "/home/schen3/xtb/bin/xtb"
SOLV = "dmso"
TMP = "/scratch-shared/schen3/tmp_diag"

ALD = [
    "C1=CC=C2C=C3C(=CC2=C1)C(=O)C(=CO3)C=O",   # fused, v1cons +5.6 -> v2 +1.3
    "CC(=O)OC1=C(NC(=C1)C2=CC=CC=C2)C=O",      # acetoxy pyrrole, +11.8 -> +8.3
]

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
        if m is None: return None
        rdDetermineBonds.DetermineConnectivity(m, charge=0)
        return topo(m)
    except Exception:
        return None

def summarize(tag, confs, ref):
    if not confs:
        print(f"  {tag:16s}: NO CONFORMERS"); return
    rows = [(E, xyz_topo(xyz) != ref) for xyz, E in confs]
    nbroken = sum(b for _, b in rows)
    loE, lo_broken = rows[0]
    intactE = next((E for E, b in rows if not b), None)
    gap = (intactE - loE) * 627.509 if (lo_broken and intactE is not None) else None
    print(f"  {tag:16s}: n={len(confs):2d} broken={nbroken:2d} "
          f"lowest_broken={lo_broken!s:5s} loE={loE:.5f}"
          + (f"  (broken min {gap:+.1f} kcal below lowest INTACT -> spurious)" if gap else ""))

def run(ald):
    bz = Th._make_benzoin_smiles(ald)
    ref = ref_topo(bz)
    print(f"\n=== {ald}\n    benzoin {bz}  ref_topo bonds={ref[0]} frags={ref[1]}")
    with tempfile.TemporaryDirectory(dir=TMP) as wd:
        wd = Path(wd)
        t = time.time(); cf = v2.rank_conformers_funnel_v2(bz, wd/"v2", XTB, solvent=SOLV, l10=20)
        summarize(f"funnel_v2 [{time.time()-t:.0f}s]", cf, ref)
        for meth in ("gfnff", "gfn2//gfnff"):
            t = time.time()
            cc = cr.rank_conformers_crest(bz, wd/("crest_"+meth.replace("/","_")), XTB,
                                          solvent=SOLV, cores=8, method=meth, l10=20)
            summarize(f"crest_{meth} [{time.time()-t:.0f}s]", cc, ref)

if __name__ == "__main__":
    Path(TMP).mkdir(parents=True, exist_ok=True)
    for a in ALD:
        try: run(a)
        except Exception:
            import traceback; traceback.print_exc()
