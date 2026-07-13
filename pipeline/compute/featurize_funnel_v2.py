#!/usr/bin/env python3
"""
ROBUST-funnel (v2) featurization variant — swaps the conformer engine to the
reproducible, RMSD-pruned, denser-sampled `conf_funnel_v2.rank_conformers_funnel_v2`
to kill the conformer-SEARCH stochasticity (~1.4 kcal scatter, ~4% 6–15 kcal
failures) that the K-convergence study traced as the real label-noise source.

Identical CLI to featurize.py; DFT level unchanged (--sp-method r2SCAN-3c).
SLURM:  FEATURIZE_SCRIPT=.../featurize_funnel_v2.py sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import thermo_orca as Th
import conf_funnel_v2

# featurize_one looks up Th._rank_conformers at call time, so this redirects both
# the aldehyde and benzoin conformer searches to the robust funnel — no edits to
# featurize.py / thermo_orca.py / conf_funnel.py.
Th._rank_conformers = conf_funnel_v2.rank_conformers_funnel_v2

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
