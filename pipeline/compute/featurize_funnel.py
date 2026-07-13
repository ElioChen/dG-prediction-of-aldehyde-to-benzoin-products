#!/usr/bin/env python3
"""
FRUST-funnel VARIANT of the featurization pipeline — a separate, frozen version
for side-by-side comparison against the canonical `featurize.py`. It reuses every
bit of `featurize.py` unchanged and only swaps the conformer engine from the legacy
`thermo_orca._rank_conformers` (GFN2-opt on `_auto_nconfs` conformers) to the FRUST
funnel in `conf_funnel.rank_conformers_funnel` (ETKDG 50/200/300 → GFN-FF → GFN2 SP
→ L10 → GFN2 opt). DFT level is unchanged (pass --sp-method r2SCAN-3c as usual).

Usage (identical CLI to featurize.py):
  python featurize_funnel.py --input rows.csv --output out/features.csv \
      --sp-method r2SCAN-3c --ensemble-k 3 ...
Or via SLURM: FEATURIZE_SCRIPT=.../featurize_funnel.py sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import thermo_orca as Th
import conf_funnel

# Swap the conformer ranker on the module object; featurize_one looks up
# Th._rank_conformers at call time, so this redirects both the aldehyde and the
# benzoin conformer searches to the funnel — without editing featurize.py.
Th._rank_conformers = conf_funnel.rank_conformers_funnel

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
