#!/usr/bin/env python3
"""Robust-funnel v2 with L20 keep-list — validation pair for featurize_funnel_v2.py.

Because v2's embedding is deterministic, v2-K10 and v2-K20 share an identical
conformer pool (the lowest 10 are common to both), so v2-K10 vs v2-K20 is a clean
ensemble-size convergence test with the search-stochasticity confound removed.
One-off validation; canonical modules untouched.
"""
from __future__ import annotations

import functools

import thermo_orca as Th
import conf_funnel_v2

Th._rank_conformers = functools.partial(conf_funnel_v2.rank_conformers_funnel_v2, l10=20)

import featurize  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(featurize.main())
