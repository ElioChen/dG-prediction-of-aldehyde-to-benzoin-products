#!/usr/bin/env python3
"""
Funnel-v3 featurization variant — funnel v2 + topology guard
(`conf_funnel_v3.rank_conformers_funnel_v3`). Drops broken-connectivity conformers
before the Boltzmann label, fixing the funnel's only catastrophic failure mode at ~zero
cost while keeping its dense deterministic sampling. See conf_funnel_v3.py.

Identical CLI to featurize.py; DFT level unchanged (--sp-method r2SCAN-3c).
SLURM:  FEATURIZE_SCRIPT=.../featurize_funnel_v3.py sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import thermo_orca as Th
import conf_funnel_v3

Th._rank_conformers = conf_funnel_v3.rank_conformers_funnel_v3

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
