#!/usr/bin/env python3
"""
FRUST-style conformer funnel (Nielsen, Rasmussen, Jensen — chemrxiv.15003686, Fig. 2),
as a DROP-IN replacement for thermo_orca._rank_conformers. Kept in a separate module
so the canonical thermo_orca.py / featurize.py stay pristine and the two conformer
strategies can be run and compared side-by-side. DFT level is unchanged (r2SCAN-3c).

Funnel (vs the legacy `_rank_conformers`, which GFN2-optimises `_auto_nconfs`
conformers — as few as 1 for ≤3 rotatable bonds):
  1. ETKDG embed 50 / 200 / 300 conformers for ≤7 / 8–12 / >12 rotatable bonds
  2. GFN-FF --opt on ALL of them          (cheap prescreen)
  3. GFN2-xTB single-point on each        (cheap ranking)
  4. keep the 10 lowest (L10)
  5. GFN2-xTB --opt tight on the L10       (the only expensive xTB step)
Returns [(opt_xyz, E_gfn2)] sorted by energy — identical interface to the legacy
ranker, so downstream (top-K Boltzmann ensemble + r2SCAN-3c ΔG) is unchanged.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Shared low-level helpers live in the canonical module (untouched).
from thermo_orca import (_mol_rotbonds, _parse_xtb_energy, _run_xtb,
                         _smiles_to_xyz_conformers, _xtb_opt_energy)


def _frust_nconfs(rotbonds: int) -> int:
    if rotbonds <= 7:
        return 50
    if rotbonds <= 12:
        return 200
    return 300


def _xtb_gfnff_opt(xyz_str: str, work_dir: Path, xtb_bin: str,
                   charge: int = 0, solvent: str = "", cores: int = 1):
    """Cheap GFN-FF geometry optimisation. Returns (opt_xyz, E_gfnff)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [xtb_bin, "input.xyz", "--gfnff", "--opt", "loose",
           "--charge", str(charge), "--norestart", "--parallel", str(cores)]
    if solvent:
        cmd += ["--alpb", solvent]
    r = _run_xtb(cmd, work_dir, timeout=300)
    opt = work_dir / "xtbopt.xyz"
    if not opt.exists():
        return None, None
    return opt.read_text(encoding="utf-8"), (_parse_xtb_energy(r.stdout) if r else None)


def _xtb_gfn2_sp(xyz_str: str, work_dir: Path, xtb_bin: str,
                 charge: int = 0, solvent: str = "", cores: int = 1):
    """GFN2-xTB single-point energy (no optimisation). Returns E or None."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [xtb_bin, "input.xyz", "--gfn", "2", "--sp",
           "--charge", str(charge), "--norestart", "--parallel", str(cores)]
    if solvent:
        cmd += ["--alpb", solvent]
    r = _run_xtb(cmd, work_dir, timeout=300)
    return _parse_xtb_energy(r.stdout) if r else None


def rank_conformers_funnel(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int = 0, title: str = "", solvent: str = "",
    cores: int = 1, workers: int = 1, l10: int = 10,
) -> list[tuple[str, float]]:
    """FRUST funnel ranker. `n_confs_max` is accepted for interface compatibility
    with thermo_orca._rank_conformers but ignored (count is rotbond-scaled)."""
    rb = _mol_rotbonds(smiles)
    n = _frust_nconfs(rb)
    xyz_list = _smiles_to_xyz_conformers(smiles, n, title)
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
        d = work_dir / f"ff{i:03d}"
        out = _xtb_gfnff_opt(xyz, d, xtb_bin, solvent=solvent, cores=cores)
        shutil.rmtree(d, ignore_errors=True)  # opt xyz/energy already captured in `out`
        return out
    ff = _map(_ff, enumerate(xyz_list))

    # 3+4: GFN2 single-point on each FF-opt geometry, keep the L10 lowest
    def _spj(i_xyz):
        i, xyz = i_xyz
        if xyz is None:
            return None
        d = work_dir / f"sp{i:03d}"
        E = _xtb_gfn2_sp(xyz, d, xtb_bin, solvent=solvent, cores=cores)
        shutil.rmtree(d, ignore_errors=True)  # energy already captured in `E`
        return (xyz, E) if E is not None else None
    sp = [r for r in _map(_spj, ((i, o) for i, (o, _) in enumerate(ff))) if r]
    if not sp:
        return []
    sp.sort(key=lambda t: t[1])
    keep = [xyz for xyz, _ in sp[:l10]]

    # 5: full GFN2 --opt on the L10 survivors
    def _opt(i_xyz):
        i, xyz = i_xyz
        return _xtb_opt_energy(xyz, work_dir / f"opt{i:02d}", xtb_bin,
                               solvent=solvent, cores=cores)
    res = _map(_opt, enumerate(keep))
    out = [(o, E) for o, E in res if o is not None and E is not None]
    out.sort(key=lambda t: t[1])
    return out
