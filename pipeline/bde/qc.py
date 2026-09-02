"""Shared BDE label QC, replacing the ad-hoc [20,200] kcal/mol window used in the first
pass of Phase-1 baselines. Combines a physical-plausibility floor/ceiling (BDE can't be
negative or absurdly large -- pure MAD doesn't know that) with the project's own
established MAD-based robust-outlier convention (delta_core._finalize_table's
`|corr-median| <= k_mad*MAD`), rather than a single arbitrary window.

Comparison that motivated this (2026-07-15): a pure MAD window ([20,200]-window vs
median+/-6MAD) disagreed by ~2% of rows for aldehydes and ~3% for products -- small, but
the pure-MAD window for products extended to -4.8 kcal/mol, which is unphysical for a
bond dissociation energy, confirming physical bounds are still needed alongside MAD.
"""
import pandas as pd


def norm_id(s: pd.Series) -> pd.Series:
    """Normalize an id column to clean integer strings ('2.0' -> '2'); NaN -> ''.

    The 2026-07 purge recovery re-serialized some homo_v6 label / scaffold-split /
    alfabet CSVs with float-formatted ids (`2.0`). The descriptor libraries
    (`*_all.csv`, rebuilt by assemble_homo_descriptor_libs.py from a 0-based integer
    library index) use `2`. A plain string merge of the two silently drops every
    float-id row -- inner-joining to ~0 rows. Call this on every id column right after
    read so all joins line up regardless of which file got the float formatting.
    """
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str).replace("<NA>", "")


def qc_filter(y: pd.Series, phys_min: float = 10.0, phys_max: float = 250.0,
              k_mad: float = 6.0) -> pd.Series:
    """Boolean mask: physically plausible AND within k_mad*MAD of the median."""
    phys_ok = y.between(phys_min, phys_max)
    med = y[phys_ok].median()
    mad = (y[phys_ok] - med).abs().median()
    mad_ok = (y - med).abs() <= k_mad * mad
    return phys_ok & mad_ok
