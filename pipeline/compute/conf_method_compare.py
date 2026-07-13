#!/usr/bin/env python
"""Quantify the conformer-search effect on the benzoin ΔG (legacy _rank_conformers vs
funnel_v3) for the DFT-opt pilot molecules — to decide whether the full-library DFT run
is worth switching off the legacy unguarded search.

For each molecule and each conformer method, take the lowest conformer of the aldehyde
and the benzoin product, then:
  - ohess (GFN2-xTB) -> G_xtb, E_el_xtb  => thermal = G_xtb - E_el_xtb
  - r2SCAN-3c SP on that geometry        => E_orca
  - benzoin topology check (broken connectivity?)
ΔG_xtb  = G_xtb(bz) - 2 G_xtb(ald)
ΔG_orca = (E_orca + thermal)(bz) - 2 (E_orca + thermal)(ald)   [DFT elec + xTB RRHO,
          same definition as the validation]

Output per molecule: dG_xtb / dG_orca / broken for BOTH methods, so we can see how much
funnel_v3 moves ΔG and whether it removes the broken-topology outliers.
"""
from __future__ import annotations
import argparse, csv, os, sys, time, shutil, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules["conf_funnel_v3"] = types.ModuleType("conf_funnel_v3")   # stub
import thermo_orca as T
del sys.modules["conf_funnel_v3"]
import conf_funnel_v3 as fv3                                          # real
from conf_crest import _ref_topo, _xyz_topo
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HART = T.HARTREE_TO_KCAL


def lowest_geom(ranker, smiles, wd, args, title):
    wd.mkdir(parents=True, exist_ok=True)
    ranked = ranker(smiles, wd, args.xtb_bin, n_confs_max=args.n_confs,
                    title=title, solvent=args.solvent, cores=1, workers=args.workers)
    if not ranked:
        return None
    return ranked[0][0]                       # lowest-energy conformer xyz string


def species(ranker, smiles, wd, args, title):
    xyz = lowest_geom(ranker, smiles, wd / "rank", args, title)
    if xyz is None:
        return None
    stdout, _ = T.run_ohess(xyz, wd / "ohess", args.xtb_bin, solvent=args.solvent)
    G = T.parse_xtb_G(stdout); E = T._parse_xtb_energy(stdout)
    if G is None or E is None:
        return None
    (wd / "g.xyz").write_text(xyz)
    Eo = T.calc_orca_sp(wd / "g.xyz", args.method, "", args.orca_solvent,
                        nprocs=args.orca_nprocs, maxcore_mb=args.orca_maxcore,
                        orca_bin=args.orca_bin, timeout=args.sp_timeout)
    return dict(xyz=xyz, G_xtb=G, thermal=G - E, E_orca=Eo)


def broken(smiles, xyz):
    try:
        return _xyz_topo(xyz) != _ref_topo(smiles)
    except Exception:
        return None


def run_method(ranker, ald, bz, wd, args):
    A = species(ranker, ald, wd / "ald", args, "ald")
    B = species(ranker, bz, wd / "bz", args, "bz")
    if not A or not B:
        return dict(dG_xtb=None, dG_orca=None, broken_bz=None)
    dG_xtb = (B["G_xtb"] - 2 * A["G_xtb"]) * HART
    dG_orca = None
    if A["E_orca"] is not None and B["E_orca"] is not None:
        Gbz = B["E_orca"] + B["thermal"]; Gald = A["E_orca"] + A["thermal"]
        dG_orca = (Gbz - 2 * Gald) * HART
    return dict(dG_xtb=round(dG_xtb, 3),
                dG_orca=round(dG_orca, 3) if dG_orca is not None else None,
                broken_bz=broken(bz, B["xyz"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="r2SCAN-3c")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--orca-solvent", default="DMSO")
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--orca-bin", default="/home/schen3/orca/orca")
    ap.add_argument("--orca-nprocs", type=int, default=24)
    ap.add_argument("--orca-maxcore", type=int, default=1500)
    ap.add_argument("--sp-timeout", type=int, default=7200)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    if args.skip: rows = rows[args.skip:]
    if args.max: rows = rows[:args.max]
    scratch = Path(args.scratch) / f"confcmp_{os.getpid()}"
    fields = ["idx", "aldehyde_smiles", "benzoin_smiles",
              "dG_xtb_legacy", "dG_orca_legacy", "broken_legacy",
              "dG_xtb_funnelv3", "dG_orca_funnelv3", "broken_funnelv3",
              "d_dG_xtb", "d_dG_orca", "note"]
    out = open(args.output, "w", newline=""); w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader(); out.flush()

    for i, r in enumerate(rows):
        ald = r["aldehyde_smiles"].strip()
        bz = (r.get("benzoin_smiles") or "").strip() or T._make_benzoin_smiles(ald)
        rec = {"idx": r.get("idx", i), "aldehyde_smiles": ald, "benzoin_smiles": bz, "note": ""}
        t0 = time.time(); wd = scratch / f"m{i:03d}"
        try:
            leg = run_method(T._rank_conformers, ald, bz, wd / "legacy", args)
            f3 = run_method(
                lambda *a, **k: fv3.rank_conformers_funnel_v3(*a, **k),
                ald, bz, wd / "funnelv3", args)
        except Exception as e:
            rec["note"] = f"err:{e}"[:80]; w.writerow(rec); out.flush()
            shutil.rmtree(wd, ignore_errors=True); continue
        rec.update(
            dG_xtb_legacy=leg["dG_xtb"], dG_orca_legacy=leg["dG_orca"], broken_legacy=leg["broken_bz"],
            dG_xtb_funnelv3=f3["dG_xtb"], dG_orca_funnelv3=f3["dG_orca"], broken_funnelv3=f3["broken_bz"],
            d_dG_xtb=round(f3["dG_xtb"] - leg["dG_xtb"], 3) if (f3["dG_xtb"] is not None and leg["dG_xtb"] is not None) else None,
            d_dG_orca=round(f3["dG_orca"] - leg["dG_orca"], 3) if (f3["dG_orca"] is not None and leg["dG_orca"] is not None) else None,
            note=f"{time.time()-t0:.0f}s")
        w.writerow(rec); out.flush()
        print(f"[{i+1}/{len(rows)}] idx={rec['idx']} "
              f"dGxtb {leg['dG_xtb']}->{f3['dG_xtb']}  broken {leg['broken_bz']}->{f3['broken_bz']} "
              f"({rec['note']})", flush=True)
        shutil.rmtree(wd, ignore_errors=True)
    out.close(); shutil.rmtree(scratch, ignore_errors=True)
    print("DONE", args.output)


if __name__ == "__main__":
    sys.exit(main())
