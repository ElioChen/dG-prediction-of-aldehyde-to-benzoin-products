#!/usr/bin/env python3
"""K-convergence-check VARIANT of the funnel featurizer — L30.

Identical to `featurize_funnel.py` / `featurize_funnel_L20.py` except the funnel
keeps the lowest **30** GFN2-opt conformers (l10=30), so an `--ensemble-k 30`
Boltzmann average completes the K3 → K10 → K20 → K30 convergence curve in one shot.
Pairs with the K20 run to confirm whether the DFT ΔG label has plateaued (and thus
whether a full-set re-label should use K10 or higher). One-off validation path, NOT
production — the canonical `conf_funnel.py` / `featurize.py` stay untouched.

Usage / SLURM:
  FEATURIZE_SCRIPT=.../featurize_funnel_L30.py ENSEMBLE_K=30 sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import functools

import thermo_orca as Th
import conf_funnel

Th._rank_conformers = functools.partial(conf_funnel.rank_conformers_funnel, l10=30)

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
