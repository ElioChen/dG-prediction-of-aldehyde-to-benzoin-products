#!/usr/bin/env python
"""DFT-geometry benchmark: does r2SCAN-3c geometry optimisation change the benzoin
reaction ΔG beyond the r2SCAN-3c//xTB single point we already run?

For each molecule (aldehyde + its benzoin product) and each species:
  1. lowest GFN2-xTB conformer geometry  (same default path as the validation run)
  2. xTB ohess  -> G_xtb, E_el_xtb  (thermal correction = G_xtb - E_el_xtb)
  3. r2SCAN-3c SP at the xTB geometry             -> E_dft @ xTB-geom
  4. r2SCAN-3c Opt (CPCM dmso) from that geometry -> E_dft @ DFT-geom, opt xyz
  5. heavy-atom RMSD( xTB-geom , DFT-geom )

Reaction-level (ΔG = product - 2*aldehyde):
  dE_xtbgeom   = E_dft//xTB  reaction electronic ΔE
  dE_dftgeom   = E_dft//DFT  reaction electronic ΔE
  geom_term    = dE_dftgeom - dE_xtbgeom        <-- pure GEOMETRY effect on DFT energy
  dG_dft_xtbgeom = dE_xtbgeom + xTB-thermal      (= what the production //xTB ΔG is)
  dG_dft_dftgeom = dE_dftgeom + xTB-thermal      (thermal held fixed; geometry relaxed)

A small geom_term (<~2 kcal) means r2SCAN-3c//xTB is adequate and full DFT-opt is not
worth it; a large one (esp. for hypervalent-S) means those motifs need DFT geometries.

Thermal is held at xTB-RRHO on both sides so geom_term isolates the geometry's
electronic effect.  (--freq adds true DFT thermal; off by default — numerical freq is
expensive.)
"""
from __future__ import annotations
import argparse, csv, os, re, sys, time, shutil, types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# thermo_orca has a circular import via conf_funnel_v3 that only resolves when it
# runs as __main__. We use the DEFAULT conformer path (funnel_v3 never called), so
# stub conf_funnel_v3 to break the cycle for a clean `import thermo_orca`.
sys.modules.setdefault("conf_funnel_v3", types.ModuleType("conf_funnel_v3"))
import thermo_orca as T

HART2KCAL = T.HARTREE_TO_KCAL


def _read_xyz_coords(path_or_str, as_str=False):
    txt = Path(path_or_str).read_text() if not as_str else path_or_str
    lines = txt.splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        syms.append(p[0]); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.array(xyz)


_RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07,
             S=1.05, Cl=1.02, Se=1.20, Br=1.20, I=1.39)


def _hbond_HO(xyz_str):
    """Shortest intramolecular hydroxyl O-H...O=C contact (Å) perceived from geometry;
    np.nan if no hydroxyl+carbonyl pair. Used to test if xTB vs DFT place the
    alpha-hydroxyketone H-bond differently."""
    try:
        sym, X = _read_xyz_coords(xyz_str, as_str=True)
    except Exception:
        return float("nan")
    n = len(sym); D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    nb = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cut = (_RCOV.get(sym[i], 0.77) + _RCOV.get(sym[j], 0.77)) * 1.3
            if 0.4 < D[i, j] < cut:
                nb[i].append(j); nb[j].append(i)
    hydroxyl = []; carbonyl = []
    for i, s in enumerate(sym):
        if s != "O":
            continue
        hs = [j for j in nb[i] if sym[j] == "H"]; cs = [j for j in nb[i] if sym[j] == "C"]
        if len(hs) == 1 and len(cs) == 1:
            hydroxyl.append(hs[0])
        elif len(nb[i]) == 1 and sym[nb[i][0]] == "C" and D[i, nb[i][0]] < 1.30:
            carbonyl.append(i)
    best = float("nan")
    for h in hydroxyl:
        for oc in carbonyl:
            d = D[h, oc]
            if np.isnan(best) or d < best:
                best = float(d)
    return best


def _heavy_rmsd(symsA, A, symsB, B):
    """Kabsch heavy-atom RMSD; assumes identical atom ordering (ORCA preserves it)."""
    keep = [i for i, s in enumerate(symsA) if s != "H"]
    if len(keep) != sum(1 for s in symsB if s != "H") or not keep:
        return None
    P = A[keep] - A[keep].mean(0)
    Q = B[[i for i, s in enumerate(symsB) if s != "H"]]
    Q = Q - Q.mean(0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pr = (R @ P.T).T
    return float(np.sqrt(np.mean(np.sum((Pr - Q) ** 2, axis=1))))


def _orca_opt(xyz_str, work_dir: Path, method, orca_solvent, orca_bin,
              nprocs, maxcore, do_freq, timeout):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "mol.xyz").write_text(xyz_str)
    task = "Opt Freq" if do_freq else "Opt"
    solv = f"CPCM({orca_solvent})" if orca_solvent else ""
    head = " ".join(x for x in [method, task, "TightSCF", "NoMOPrint", solv] if x)
    inp = (f"! {head}\n\n%pal nprocs {nprocs} end\n\n%maxcore {maxcore}\n\n"
           f"%geom MaxIter 120 end\n\n* xyzfile 0 1 mol.xyz\n")
    (work_dir / "input.inp").write_text(inp)
    env = {**os.environ,
           "PATH": f"{Path(orca_bin).parent}:{os.environ.get('PATH','')}",
           "LD_LIBRARY_PATH": f"{Path(orca_bin).parent/'lib'}:{os.environ.get('LD_LIBRARY_PATH','')}"}
    import subprocess
    E = G = None; opt_xyz = None
    try:
        r = subprocess.run([orca_bin, "input.inp"], cwd=str(work_dir),
                           capture_output=True, text=True, timeout=timeout, env=env)
        out = r.stdout + r.stderr
        (work_dir / "input.out").write_text(out)
        E = T._parse_orca_energy(out)            # FINAL SINGLE POINT ENERGY (last = opt geom)
        m = re.search(r"Final Gibbs free energy\s*\.*\s*(-?\d+\.\d+)", out)
        if m:
            G = float(m.group(1))
        of = work_dir / "input.xyz"               # ORCA writes optimised geometry here
        if of.exists():
            opt_xyz = of.read_text()
        conv = "THE OPTIMIZATION HAS CONVERGED" in out
        return dict(E=E, G=G, opt_xyz=opt_xyz, converged=conv)
    except subprocess.TimeoutExpired:
        return dict(E=None, G=None, opt_xyz=None, converged=False, timeout=True)
    finally:
        for pat in ["*.gbw", "*.tmp", "*.densities", "*.ges", "*_ges", "*.cpcm*", "*.bas*"]:
            for f in work_dir.glob(pat):
                try: f.unlink()
                except Exception: pass


def species_calc(smiles, title, wd: Path, args):
    """Return dict of energies for one species, or None on failure."""
    ranked = T._rank_conformers(smiles, wd / "rank", args.xtb_bin,
                                n_confs_max=args.n_confs, title=title,
                                solvent=args.solvent, cores=1, workers=args.workers)
    if not ranked:
        return None
    xtb_xyz, E_xtb_el = ranked[0]                 # lowest xTB conformer geometry
    # xTB thermal (ohess) at that geometry
    stdout, _ = T.run_ohess(xtb_xyz, wd / "ohess", args.xtb_bin, solvent=args.solvent)
    G_xtb = T.parse_xtb_G(stdout)
    E_xtb = T._parse_xtb_energy(stdout)
    thermal = (G_xtb - E_xtb) if (G_xtb is not None and E_xtb is not None) else None
    # write xTB geom to a file for SP
    (wd / "xtbgeom.xyz").write_text(xtb_xyz)
    E_dft_xtbgeom = T.calc_orca_sp(wd / "xtbgeom.xyz", args.method, "", args.orca_solvent,
                                   nprocs=args.orca_nprocs, maxcore_mb=args.orca_maxcore,
                                   orca_bin=args.orca_bin, timeout=args.sp_timeout)
    opt = _orca_opt(xtb_xyz, wd / "opt", args.method, args.orca_solvent, args.orca_bin,
                    args.orca_nprocs, args.orca_maxcore, args.freq, args.opt_timeout)
    odir = getattr(args, "orca_out_dir", "")
    if odir:
        src = wd / "opt" / "input.out"
        if src.exists():
            Path(odir).mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(src, Path(odir) / f"{wd.parent.name}_{wd.name}_opt.out")
            except Exception:
                pass
    rmsd = None
    hb_dft = float("nan")
    if opt["opt_xyz"]:
        sA, A = _read_xyz_coords(xtb_xyz, as_str=True)
        sB, B = _read_xyz_coords(opt["opt_xyz"], as_str=True)
        rmsd = _heavy_rmsd(sA, A, sB, B)
        hb_dft = _hbond_HO(opt["opt_xyz"])
    return dict(G_xtb=G_xtb, E_xtb_el=E_xtb, thermal=thermal,
                E_dft_xtbgeom=E_dft_xtbgeom, E_dft_dftgeom=opt["E"],
                G_dft_full=opt["G"], rmsd=rmsd, converged=opt["converged"],
                hbond_xtb=_hbond_HO(xtb_xyz), hbond_dft=hb_dft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with aldehyde_smiles[,benzoin_smiles]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="r2SCAN-3c")
    ap.add_argument("--solvent", default="dmso", help="xTB ALPB solvent")
    ap.add_argument("--orca-solvent", default="DMSO", help="ORCA CPCM solvent")
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--orca-bin", default="/home/schen3/orca/orca")
    ap.add_argument("--orca-nprocs", type=int, default=8)
    ap.add_argument("--orca-maxcore", type=int, default=2500)
    ap.add_argument("--sp-timeout", type=int, default=7200)
    ap.add_argument("--opt-timeout", type=int, default=36000)
    ap.add_argument("--freq", action="store_true", help="also DFT freq (true DFT thermal; slow)")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--orca-out-dir", default="", help="persist each ORCA opt input.out here (for SCF-failure diagnosis)")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    if args.skip:
        rows = rows[args.skip:]
    if args.max:
        rows = rows[:args.max]
    scratch = Path(args.scratch) / f"dftopt_{os.getpid()}"
    fields = ["idx", "aldehyde_smiles", "benzoin_smiles",
              "dE_xtbgeom_kcal", "dE_dftgeom_kcal", "geom_term_kcal",
              "dG_dft_xtbgeom_kcal", "dG_dft_dftgeom_kcal", "dG_dft_full_kcal",
              "rmsd_ald", "rmsd_bz", "bz_hbond_xtb", "bz_hbond_dft", "bz_hbond_delta",
              "ald_conv", "bz_conv", "note"]
    out_f = open(args.output, "w", newline="")
    w = csv.DictWriter(out_f, fieldnames=fields); w.writeheader(); out_f.flush()

    for i, r in enumerate(rows):
        ald = r["aldehyde_smiles"].strip()
        bz = (r.get("benzoin_smiles") or "").strip() or T._make_benzoin_smiles(ald)
        idx = r.get("idx", i)
        t0 = time.time()
        rec = {"idx": idx, "aldehyde_smiles": ald, "benzoin_smiles": bz, "note": ""}
        if not bz:
            rec["note"] = "no_benzoin"; w.writerow(rec); out_f.flush(); continue
        wd = scratch / f"mol_{i:04d}"
        try:
            A = species_calc(ald, "ald", wd / "ald", args)
            B = species_calc(bz, "bz", wd / "bz", args)
        except Exception as e:
            rec["note"] = f"err:{e}"[:80]; w.writerow(rec); out_f.flush()
            shutil.rmtree(wd, ignore_errors=True); continue
        if not A or not B:
            rec["note"] = "species_fail"; w.writerow(rec); out_f.flush()
            shutil.rmtree(wd, ignore_errors=True); continue

        def dd(field):
            if A[field] is None or B[field] is None:
                return None
            return (B[field] - 2 * A[field]) * HART2KCAL
        dE_x = dd("E_dft_xtbgeom"); dE_d = dd("E_dft_dftgeom")
        # thermal in Hartree applied identically on both geometry choices
        th = None
        if A["thermal"] is not None and B["thermal"] is not None:
            th = (B["thermal"] - 2 * A["thermal"]) * HART2KCAL
        rec.update(
            dE_xtbgeom_kcal=round(dE_x, 3) if dE_x is not None else None,
            dE_dftgeom_kcal=round(dE_d, 3) if dE_d is not None else None,
            geom_term_kcal=round(dE_d - dE_x, 3) if (dE_x is not None and dE_d is not None) else None,
            dG_dft_xtbgeom_kcal=round(dE_x + th, 3) if (dE_x is not None and th is not None) else None,
            dG_dft_dftgeom_kcal=round(dE_d + th, 3) if (dE_d is not None and th is not None) else None,
            dG_dft_full_kcal=round(dd("G_dft_full"), 3) if dd("G_dft_full") is not None else None,
            rmsd_ald=round(A["rmsd"], 3) if A["rmsd"] is not None else None,
            rmsd_bz=round(B["rmsd"], 3) if B["rmsd"] is not None else None,
            bz_hbond_xtb=round(B["hbond_xtb"], 3) if B["hbond_xtb"] == B["hbond_xtb"] else None,
            bz_hbond_dft=round(B["hbond_dft"], 3) if B["hbond_dft"] == B["hbond_dft"] else None,
            bz_hbond_delta=round(B["hbond_dft"] - B["hbond_xtb"], 3)
                if (B["hbond_dft"] == B["hbond_dft"] and B["hbond_xtb"] == B["hbond_xtb"]) else None,
            ald_conv=A["converged"], bz_conv=B["converged"],
            note=f"{time.time()-t0:.0f}s",
        )
        w.writerow(rec); out_f.flush()
        print(f"[{i+1}/{len(rows)}] idx={idx} geom_term={rec['geom_term_kcal']} "
              f"dG//xtb={rec['dG_dft_xtbgeom_kcal']} dG//dft={rec['dG_dft_dftgeom_kcal']} "
              f"({rec['note']})", flush=True)
        shutil.rmtree(wd, ignore_errors=True)
    out_f.close()
    shutil.rmtree(scratch, ignore_errors=True)
    print("DONE", args.output)


if __name__ == "__main__":
    sys.exit(main())
