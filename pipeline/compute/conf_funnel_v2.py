#!/usr/bin/env python3
"""
ROBUST funnel (v2) — a reproducibility/robustness upgrade of conf_funnel.py.

The K3→K10→K20→K30 convergence study (50-molecule subset) showed the funnel's
DFT ΔG labels carry ~1.4 kcal scatter and ~4% catastrophic failures (6–15 kcal)
that are *uncorrelated with flexibility* (the worst case was a 3-rotatable-bond
molecule swinging 15.6 kcal). Diagnosis: the labels are limited by conformer-SEARCH
stochasticity, not ensemble size — the v1 funnel embeds with `numThreads=0`
(multithreaded RDKit embedding is non-deterministic even with a fixed seed) and too
few, non-deduplicated conformers, so independent runs sometimes miss the true global
minimum and a bad geometry dominates the Boltzmann average.

v2 fixes the SEARCH (DFT level r2SCAN-3c is unchanged):
  • deterministic embedding  — numThreads=1 + randomSeed=42 (fully reproducible)
  • RMSD pruning             — pruneRmsThresh so the kept conformers are DISTINCT
                               basins, not near-duplicates of one well
  • denser sampling          — ~2× the v1 conformer counts (the GFN-FF prescreen is
                               cheap; the cost is in the L10 GFN2-opt + DFT, unchanged)
Everything downstream (GFN-FF opt → GFN2 SP → keep L → GFN2 opt → top-K Boltzmann)
is identical to v1, so v1↔v2 is a clean A/B on the conformer search only.

Kept separate from conf_funnel.py / thermo_orca.py (both stay pristine) for
side-by-side comparison; use via featurize_funnel_v2.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs

# Reuse the canonical xTB helpers + the v1 funnel stages (untouched).
from thermo_orca import (_mol_rotbonds, _parse_xtb_energy, _run_xtb,  # noqa: F401
                         _xtb_opt_energy)
from conf_funnel import _xtb_gfn2_sp, _xtb_gfnff_opt


def _mol_heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0


def _frust_nconfs_v2(rotbonds: int, n_heavy: int = 0) -> int:
    """~2× v1 (50/200/300) — denser sampling to reliably hit the global basin,
    with a HEAVY-ATOM cap on top of the rotbond schedule.

    The rotbond-only count made large benzoin products grind for hours: each
    conformer costs a GFN-FF opt + (for the L survivors) a full GFN2 --opt, both of
    which scale steeply with system size, so a 600-conformer dense search on a
    ~60-heavy-atom product times out repeatedly. Big molecules already pay far more
    per conformer, so fewer conformers is the right trade — cap the dense count
    above ~55 heavy atoms to keep per-molecule wall-clock bounded.
    """
    if rotbonds <= 7:
        n = 100
    elif rotbonds <= 12:
        n = 400
    else:
        n = 600
    if n_heavy >= 70:
        n = min(n, 100)
    elif n_heavy >= 55:
        n = min(n, 200)
    return n


def _rmsd_prune_thresh(rotbonds: int) -> float:
    """Looser pruning for floppier molecules (more genuinely distinct basins)."""
    return 0.5 if rotbonds <= 7 else 0.35


def _embed_conformers_robust(smiles: str, n_confs: int, prune: float,
                             title: str = "") -> list[str]:
    """Deterministic, RMSD-pruned ETKDGv3 embed + MMFF. Single-threaded so a fixed
    seed gives byte-identical conformers run-to-run (unlike numThreads=0)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1            # reproducibility: 1 thread, fixed seed
    params.maxIterations = 1000
    params.pruneRmsThresh = prune    # drop near-duplicate basins at embed time
    n_emb = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if n_emb == 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return []
    try:
        MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=1)
    except Exception:
        pass
    xyz_list: list[str] = []
    for conf in mol.GetConformers():
        n = mol.GetNumAtoms()
        lines = [str(n), title or smiles[:60]]
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()}  {p.x:.8f}  {p.y:.8f}  {p.z:.8f}")
        xyz_list.append("\n".join(lines))
    return xyz_list


def rank_conformers_funnel_v2(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int = 0, title: str = "", solvent: str = "",
    cores: int = 1, workers: int = 1, l10: int = 10,
) -> list[tuple[str, float]]:
    """Robust funnel ranker — same interface as conf_funnel.rank_conformers_funnel.
    `n_confs_max` accepted for compatibility but ignored (count is rotbond-scaled)."""
    rb = _mol_rotbonds(smiles)
    nh = _mol_heavy_atoms(smiles)
    xyz_list = _embed_conformers_robust(smiles, _frust_nconfs_v2(rb, nh),
                                        _rmsd_prune_thresh(rb), title)
    if not xyz_list:
        return []
    w = max(1, workers)

    def _map(fn, items):
        items = list(items)
        if w > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=w) as ex:
                return list(ex.map(fn, items))
        return [fn(t) for t in items]

    # 1+2: GFN-FF optimise every conformer
    def _ff(i_xyz):
        i, xyz = i_xyz
        return _xtb_gfnff_opt(xyz, work_dir / f"ff{i:03d}", xtb_bin,
                              solvent=solvent, cores=cores)
    ff = _map(_ff, enumerate(xyz_list))

    # 3+4: GFN2 single-point on each FF-opt geometry, keep the L lowest
    def _spj(i_xyz):
        i, xyz = i_xyz
        if xyz is None:
            return None
        E = _xtb_gfn2_sp(xyz, work_dir / f"sp{i:03d}", xtb_bin,
                         solvent=solvent, cores=cores)
        return (xyz, E) if E is not None else None
    sp = [r for r in _map(_spj, ((i, o) for i, (o, _) in enumerate(ff))) if r]
    if not sp:
        return []
    sp.sort(key=lambda t: t[1])
    keep = [xyz for xyz, _ in sp[:l10]]

    # 5: full GFN2 --opt on the L survivors
    def _opt(i_xyz):
        i, xyz = i_xyz
        return _xtb_opt_energy(xyz, work_dir / f"opt{i:02d}", xtb_bin,
                               solvent=solvent, cores=cores)
    res = _map(_opt, enumerate(keep))
    out = [(o, E) for o, E in res if o is not None and E is not None]
    out.sort(key=lambda t: t[1])
    return out
