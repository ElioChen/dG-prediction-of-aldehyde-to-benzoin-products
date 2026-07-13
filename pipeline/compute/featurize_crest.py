#!/usr/bin/env python3
"""
CREST featurization variant — swaps the conformer engine to CREST metadynamics
(`conf_crest.rank_conformers_crest`) to remove the broken-connectivity poisoning that
the RDKit funnel suffers (diag_conf_connectivity.py): CREST's MTD search + internal
topology check returns only genuine conformers of the input.

Identical CLI to featurize.py; DFT level unchanged (--sp-method r2SCAN-3c).

ENV knobs (CREST manages its own parallelism, so featurize's cores/workers are NOT
forwarded as-is):
  CREST_METHOD  gfnff | gfn2 | gfn2//gfnff   (default gfn2//gfnff)
  CREST_CORES   threads for CREST --T        (default 8; set PARALLEL_JOBS=1 in SLURM
                                              so one molecule gets the whole task)

SLURM:
  CREST_METHOD=gfn2//gfnff CREST_CORES=8 \
  FEATURIZE_SCRIPT=.../featurize_crest.py sbatch submit_featurize_array.sh
"""
from __future__ import annotations

import functools
import os

import thermo_orca as Th
import conf_crest

_METHOD = os.environ.get("CREST_METHOD", "gfn2//gfnff")
_CORES = int(os.environ.get("CREST_CORES", "8"))


def _ranker(smiles, work_dir, xtb_bin, n_confs_max=0, title="", solvent="",
            cores=1, workers=1, l10=10):
    """Adapter with the Th._rank_conformers signature. CREST does its own threading,
    so we override `cores` with CREST_CORES (give it the whole task) and ignore the
    funnel's `workers` (per-conformer ThreadPool), which CREST does not use."""
    return conf_crest.rank_conformers_crest(
        smiles, work_dir, xtb_bin, n_confs_max=n_confs_max, title=title,
        solvent=solvent, cores=_CORES, workers=1, l10=l10, method=_METHOD)


# featurize_one looks up Th._rank_conformers at call time -> redirects both the
# aldehyde and benzoin conformer searches to CREST. No edits to featurize.py.
Th._rank_conformers = _ranker

import featurize  # noqa: E402  (import after the monkeypatch)

if __name__ == "__main__":
    raise SystemExit(featurize.main())
