#!/usr/bin/env python
"""One-molecule ORCA r2SCAN-3c SP timing probe: where does the time go, and does
multi-core (nprocs) help per-SP wall? Picks a ~median-size product geometry and runs the
SP at nprocs=1,4,8, keeping each ORCA .out for the timing breakdown."""
from __future__ import annotations
import sys, time, shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conf_funnel_v3  # noqa
import thermo_orca as T

man = pd.read_parquet("/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/manifest.parquet")
# pick a mid-size product by xyz atom count
def natoms(p):
    try: return int(open(p).readline().strip())
    except: return 0
man["na"] = man["prod_xyz"].map(natoms)
row = man.iloc[(man["na"] - man["na"].median()).abs().argsort().iloc[0]]
print(f"probe molecule id={row['id']} product atoms={row['na']}  xyz={row['prod_xyz']}")

base = Path("/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/timing_probe")
base.mkdir(parents=True, exist_ok=True)
for npx in (1, 4, 8):
    wd = base / f"np{npx}"; wd.mkdir(exist_ok=True)
    shutil.copy(row["prod_xyz"], wd / "mol.xyz")
    t0 = time.time()
    E = T.calc_orca_sp(wd / "mol.xyz", "r2SCAN-3c", "def2-mTZVP", "DMSO",
                       nprocs=npx, maxcore_mb=3000, orca_bin="/home/schen3/orca/orca")
    dt = time.time() - t0
    print(f"nprocs={npx}: wall={dt:.0f}s ({dt/60:.1f} min)  E={E}")
print("ORCA .out files kept under", base, "(grep 'TOTAL RUN TIME', 'SCF ITERATIONS', 'GEOMETRY')")
