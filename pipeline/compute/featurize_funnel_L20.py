#!/usr/bin/env python3
"""K-convergence-check VARIANT of the funnel featurizer.

Identical to `featurize_funnel.py` except the funnel keeps the lowest **20**
GFN2-opt conformers (l10=20) instead of 10, so an `--ensemble-k 20` Boltzmann
average can probe whether the K10 DFT ΔG label is converged. The K3→K10 shift was
large (+1.07 kcal systematic, 1.35 mean abs over 50 molecules); this run answers
whether K10 itself has plateaued or is still moving. One-off validation path, NOT
production — keep the canonical `conf_funnel.py` / `featurize.py` untouched.

Usage (same CLI as featurize.py):
  python featurize_funnel_L20.py --input rows.csv --output out/features.csv \
      --sp-method r2SCAN-3c --ensemble-k 20 ...
Or via SLURM:
  FEATURIZE_SCRIPT=.../featurize_funnel_L20.py ENSEMBLE_K=20 sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import functools

import thermo_orca as Th
import conf_funnel

# Same monkeypatch as featurize_funnel.py, but bind l10=20 so up to 20 conformers
# survive the funnel and are available to the Boltzmann ensemble. thermo_orca calls
# Th._rank_conformers(smiles, work_dir, xtb_bin, n_confs_max, label, solvent=...);
# the partial injects l10=20 without disturbing that positional interface.
Th._rank_conformers = functools.partial(conf_funnel.rank_conformers_funnel, l10=20)

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
