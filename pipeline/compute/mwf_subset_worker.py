#!/usr/bin/env python
"""Compute PRODUCT ADCH/QTAIM (the 16 empty MULTIWFN cols) on the SAVED funnel_v3 geometry
(no conformer search). Reuses featurize_product.find_benzoin_core + calc_multiwfn_product +
_adch_fukui_product. For the ADCH/QTAIM validation subset.
  python mwf_subset_worker.py --subset s.csv --skip 0 --max 50 --out chunk.csv --scratch /scratch-local/...
"""
import argparse, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/scratch-shared/schen3/benzoin-dg/pipeline/compute")
import featurize_product as FP
A = FP.A
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
MWF = os.environ.get("MWF_BIN", "/home/schen3/mutiwfn/Multiwfn_noGUI")

def parse_xyz(path):
    txt = Path(path).read_text()
    ls = txt.splitlines(); n = int(ls[0].split()[0])
    sym, crd = [], []
    for l in ls[2:2 + n]:
        p = l.split(); sym.append(p[0]); crd.append([float(p[1]), float(p[2]), float(p[3])])
    return txt, sym, np.array(crd, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True); ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=50); ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    df = pd.read_csv(a.subset).iloc[a.skip:a.skip + a.max]
    wroot = Path(a.scratch) / f"mwf_{a.skip}"; wroot.mkdir(parents=True, exist_ok=True)
    rows = []; t0 = time.time(); ok = 0
    for r in df.itertuples():
        rec = {"id": r.id, "error": None}
        wd = wroot / f"m{r.id}"; wd.mkdir(exist_ok=True)
        # ---- product ADCH/QTAIM (16) ----
        try:
            xyz_str, sym, crd = parse_xyz(r.prod_xyz)
            core = FP.find_benzoin_core(sym, crd)
            if core is None:
                rec["error"] = "no_core"
            else:
                rec.update(FP.calc_multiwfn_product(xyz_str, sym, crd, core, XTB, MWF, wd, f"p{r.id}"))
                rec.update(FP._adch_fukui_product(xyz_str, sym, crd, core, XTB, MWF, wd, f"f{r.id}"))
        except Exception as e:
            rec["error"] = (rec["error"] or "") + f"|prod:{str(e)[:50]}"
        # ---- reactant(aldehyde) ADCH/QTAIM (prefixed ald_) ----
        try:
            if isinstance(r.ald_xyz, str) and os.path.exists(r.ald_xyz):
                axyz, asym, acrd = parse_xyz(r.ald_xyz)
                ad = A.calc_multiwfn(axyz, asym, acrd, XTB, MWF, wd, f"a{r.id}")
                for k, v in ad.items():
                    if k.startswith("adch") or k.startswith("qtaim"):
                        rec[f"ald_{k}"] = v
        except Exception as e:
            rec["error"] = (rec["error"] or "") + f"|ald:{str(e)[:50]}"
        shutil.rmtree(wd, ignore_errors=True)
        ok += int(rec["error"] is None or "prod" not in str(rec["error"]))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    print(f"wrote {a.out}  ok={ok}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
