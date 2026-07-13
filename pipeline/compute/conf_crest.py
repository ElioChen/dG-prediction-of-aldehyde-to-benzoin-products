#!/usr/bin/env python3
"""
CREST conformer search — a drop-in alternative to conf_funnel{,_v2}.rank_conformers_*.

Motivation. The RDKit-funnel search (ETKDG embed -> GFN-FF opt -> GFN2 SP/opt)
is fast but fragile: the diag (diag_conf_connectivity.py) showed the v2 funnel's
denser GFN-FF prescreen can relax a conformer into a BROKEN-connectivity structure
(bond formed/broken or fragment) whose spuriously low energy then dominates the
Boltzmann average and poisons the DFT ΔG label. CREST (Grimme) does metadynamics-
based sampling and applies an internal topology check that discards structures whose
connectivity changed during the run, so the ensemble it returns is — by construction
— made of genuine conformers of the input molecule. That is exactly the failure mode
we want to remove from the labels.

Method choices (`method=`):
  • "gfnff"        — GFN-FF for the whole CREST search. Cheapest (~8-11 min/mol, 8
                     cores on a floppy benzoin). Energies are GFN-FF -> only use for
                     SEARCH; re-rank/Boltzmann should be done at GFN2/DFT downstream.
  • "gfn2"         — GFN2-xTB for the whole CREST search. Best xTB-level energies,
                     slowest (native, via the tblite calculator).
  • "gfn2//gfnff"  — composite (DEFAULT): CREST GFN-FF metadynamics sampling, then a
                     GFN2 `--opt tight` (xtb) on the top-L ensemble members for the
                     final energies. Returned energies are GFN2, so it is a clean
                     apples-to-apples swap for the funnel (which Boltzmann-weights on
                     GFN2), AND it is ~60x cheaper on the GFN2 stage than CREST's own
                     multilevel (which optimises ALL trajectory snapshots at GFN2).

NOTE on the native composite. CREST 3.0.2 (new tblite calculator) cannot do this on
the command line: the legacy `-gfn2//gfnff` flag errors ("not yet available with new
calculator"), and the TOML multilevel route (two `[[calculation.level]]` blocks +
`[dynamics] active=[1]`, runtype imtd-gc) SEGFAULTS in this static GNU build during
the "Multilevel Ensemble Optimization" of all ~600-700 snapshots at GFN2 ("BLAS: Bad
memory unallocation"), reproducibly across multilevelopt true/false, optlev tight/
normal, and unlimited stack. So we do the GFN2 leg ourselves with
thermo_orca._xtb_opt_energy — identical to the funnel's expensive final step, but on
CREST's robust (topology-checked, deduplicated) ensemble instead of RDKit's fragile
one, and only on the top-L conformers. (The Intel CREST build may avoid the BLAS bug
if exhaustive GFN2-on-all-snapshots is ever wanted.)

Interface mirrors conf_funnel.rank_conformers_funnel: returns [(opt_xyz, E_Eh)]
sorted ascending, so the downstream top-K Boltzmann + r2SCAN-3c ΔG is unchanged.
`n_confs_max`/`workers` are accepted for compatibility but CREST manages its own
sampling and parallelism (`cores` -> --T). `l10` caps the returned ensemble to the
L lowest (mirrors the funnel keeping L survivors for the expensive DFT step).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

# GFN2 final-leg of the composite: reuse the funnel's tight GFN2 optimiser verbatim.
from thermo_orca import _xtb_opt_energy

CREST_BIN = "/home/schen3/crest/crest/crest"
XTB_DIR = "/home/schen3/xtb/bin"

# CREST-native search levels. The composite "gfn2//gfnff" is NOT here: v3.0.2 rejects
# the native flag, so it samples with "gfnff" then re-optimises at GFN2 in xtb (below).
_NATIVE_FLAGS = {
    "gfnff": ["--gfnff"],
    "gfn2": ["--gfn2"],
}
_METHODS = (*_NATIVE_FLAGS, "gfn2//gfnff")


def _embed_start_xyz(smiles: str, title: str = "") -> str | None:
    """One deterministic ETKDGv3+MMFF starting geometry for CREST to sample from."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    # Escalating embed attempts: plain ETKDGv3 fails on some strained/caged systems,
    # so fall back to more iterations and random-coordinate initialisation before
    # giving up (this is what `ald_embed_failed` was hitting on 2/80 molecules).
    ok = False
    for seed in (42, 1, 7):
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        p.numThreads = 1
        p.maxIterations = 2000
        if AllChem.EmbedMolecule(mol, p) == 0:
            ok = True
            break
    if not ok:
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        p.numThreads = 1
        p.maxIterations = 4000
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                return None
    try:
        MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), title or smiles[:60]]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()}  {p.x:.8f}  {p.y:.8f}  {p.z:.8f}")
    return "\n".join(lines)


def _ref_topo(smiles: str):
    """Heavy-atom topology fingerprint (n_bonds, n_frags, sorted degree seq); bond-order
    independent, so it only changes if a bond actually forms/breaks or fragments."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return _topo(m)


def _topo(mol):
    rw = Chem.RWMol(mol)
    for a in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1),
                    reverse=True):
        rw.RemoveAtom(a)
    m = rw.GetMol()
    deg = tuple(sorted(a.GetDegree() for a in m.GetAtoms()))
    return (m.GetNumBonds(), len(Chem.GetMolFrags(m)), deg)


def _xyz_topo(xyz_block: str):
    try:
        m = Chem.MolFromXYZBlock(xyz_block)
        if m is None:
            return None
        rdDetermineBonds.DetermineConnectivity(m, charge=0)
        return _topo(m)
    except Exception:
        return None


def _parse_crest_ensemble(path: Path, natoms: int) -> list[tuple[str, float]]:
    """Parse crest_conformers.xyz -> [(xyz_block, E_Eh)], already energy-sorted by CREST.
    Each block is natoms+2 lines; the comment line is the total energy in Hartree."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[tuple[str, float]] = []
    i, stride = 0, natoms + 2
    while i + stride <= len(lines):
        block = lines[i:i + stride]
        try:
            E = float(block[1].split()[0])
        except (ValueError, IndexError):
            E = None
        if E is not None:
            out.append(("\n".join(block) + "\n", E))
        i += stride
    return out


def _crest_sample(start_xyz: str, work_dir: Path, native: str, solvent: str,
                  cores: int, ewin: float, quick: bool, crest_bin: str,
                  timeout: int) -> list[tuple[str, float]]:
    """Run one CREST search at a native level -> energy-sorted [(xyz, E_Eh)]."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "start.xyz").write_text(start_xyz, encoding="utf-8")
    natoms = int(start_xyz.split("\n", 1)[0])
    cmd = [crest_bin, "start.xyz", *_NATIVE_FLAGS[native],
           "--T", str(max(1, cores)), "--ewin", str(ewin)]
    if solvent:
        cmd += ["--alpb", solvent]
    if quick:
        cmd += ["--quick"]
    # CREST 3.0 GFN methods are built-in (tblite/gfnff); keep xtb on PATH defensively.
    env = dict(os.environ)
    env["PATH"] = XTB_DIR + os.pathsep + env.get("PATH", "")
    env.setdefault("OMP_NUM_THREADS", str(max(1, cores)))
    try:
        subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True,
                       text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, Exception):
        return []
    return _parse_crest_ensemble(work_dir / "crest_conformers.xyz", natoms)


def rank_conformers_crest(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int = 0, title: str = "", solvent: str = "",
    cores: int = 1, workers: int = 1, l10: int = 10,
    *, method: str = "gfn2//gfnff", ewin: float = 6.0,
    quick: bool = False, crest_bin: str = CREST_BIN, timeout: int = 5400,
) -> list[tuple[str, float]]:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    start = _embed_start_xyz(smiles, title)
    if start is None:
        return []

    ref = _ref_topo(smiles)
    native = "gfnff" if method == "gfn2//gfnff" else method
    ens = _crest_sample(start, work_dir / "crest", native, solvent,
                        cores, ewin, quick, crest_bin, timeout)
    if not ens:
        return []

    # Topology guard: CREST already discards connectivity-changed structures, but verify
    # against the input graph so a perception edge-case can never poison the label.
    clean = [(xyz, E) for xyz, E in ens if _xyz_topo(xyz) == ref]
    if not clean:                      # perception failed on all -> trust CREST's own check
        clean = ens
    clean.sort(key=lambda t: t[1])

    if method != "gfn2//gfnff":
        return clean[:l10] if l10 and l10 > 0 else clean

    # Composite final leg: GFN2 --opt tight (xtb) on the L lowest GFN-FF conformers,
    # re-check topology, re-rank by GFN2 energy. This is the funnel's expensive step,
    # applied to CREST's robust ensemble.
    keep = clean[:l10] if l10 and l10 > 0 else clean
    out: list[tuple[str, float]] = []
    for i, (xyz, _) in enumerate(keep):
        opt_xyz, E = _xtb_opt_energy(xyz, work_dir / f"gfn2opt{i:02d}", xtb_bin,
                                     solvent=solvent, cores=cores)
        if opt_xyz is None or E is None:
            continue
        if _xyz_topo(opt_xyz) == ref:
            out.append((opt_xyz, E))
    out.sort(key=lambda t: t[1])
    return out
