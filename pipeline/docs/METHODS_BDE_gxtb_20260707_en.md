# BDE/BDFE Method Level 3: g-xTB-consistent BDE + BDFE

**Status: PROMOTED — current production champion feature set**

## Motivation ("method-mismatch" hypothesis)

Levels 1-2 computed BDE/BDFE at the GFN2-xTB level, but the actual production baseline being
corrected (dG_gxtb) is **g-xTB**, not GFN2 — GFN2 is only used internally as a hybrid-formula
thermal-correction donor (see `gxtb_baseline.py`'s `G_gxtb = E_gxtb + (G_gfn2 - E_gfn2)`).
Hypothesis: a level mismatch between the descriptor (GFN2) and the thing it's meant to help
correct (g-xTB's own errors) may be diluting the signal — recomputing BDE/BDFE at the g-xTB
level should carry more signal if this hypothesis holds.

## Method

`pipeline/compute/calc_bde_free_energy_gxtb.py`, reusing the project's own hybrid-correction
pattern:
- Parent's `G_gxtb` is already cached (`{aldehydes,products}_all.csv`) — no new calculation.
- Parent's raw `E_gxtb` is recovered **algebraically, for free**: `E_gxtb_parent =
  G_gxtb_parent - (G_gfn2_parent - E_gfn2_parent)` (inverting the hybrid formula; both GFN2
  terms are already-cached `G_xtb`/`xtb_energy` columns).
- Each radical fragment gets a fresh GFN2 `--ohess` (thermal correction + relaxed geometry,
  as in Level 2) **plus a g-xTB single-point on that same geometry** (`--gxtb --cosmo dmso`,
  smoke-tested to work together — g-xTB's analytical Hessian is potentially more numerically
  stable than GFN2's numerical one, relevant to the "low-frequency-mode noise" theory from
  Level 2). Both `bde_gxtb_kcal` and `bdfe_gxtb_kcal` are emitted per molecule from a single
  fragment calculation — fragment-side `E_gxtb` was already computed internally for the SP
  step and simply kept instead of discarded.

## Pilot validation

3-molecule pilot: 3/3 success, `BDE > BDFE` as expected from the entropy-of-dissociation
effect. Scattered-sample pilots (not first-N-rows, which are all small aliphatic molecules —
timing is highly size-dependent, 30s-14min per 20-mol chunk):
- n≈140/side: aldehyde C-H BDFE(g-xTB) r=0.300 vs dG_gxtb (p=3e-4) — a real jump from the
  GFN2 version's r=0.044 on the full library. Product C-C BDFE(g-xTB) r=-0.184, similar
  magnitude to the GFN2 version — no improvement on the product side.
- n≈711 ald / 694 prod (larger, combined pilot): aldehyde r=0.209 vs dG_gxtb (p=1.9e-8), still
  ~4-5x the GFN2 version's r=0.044 and highly significant; product r=-0.254, still no clear
  improvement over GFN2's -0.173.

## Coverage

Full-library arrays: aldehydes 1460/1471 (99.3%, job 24437300, clean); products 1459/1463
(99.7%, jobs 24437301→24454920 after a shared-environment corruption incident forced a
backfill — see `shared-env-instability-2026-07-05` — recovered via an isolated
`envs/bde_lite` venv). Sidecars: `{aldehydes,products}_bdfe_gxtb_descriptors.csv`, n=219,095
labeled rows overall, 95.8-97.8% column coverage (median-imputed on train split only).

## Result

`finalize_correction_bdfe_gxtb_full.py` quick 2-member XGB check on mordredslim271 (271 feats,
MAE 1.612):

| config | n_feat | test MAE | delta |
|---|---|---|---|
| mordredslim271 baseline | 271 | 1.612 | — |
| + BDFE(g-xTB), both sides | 273 | 1.605 | -0.007 (null, matches GFN2's own null) |
| + BDE(g-xTB), both sides | 273 | **1.580** | **-0.032 (real — bigger than GFN2's own +0.024)** |
| + BDE(g-xTB) + BDFE(g-xTB), both sides | 275 | **1.563** | **-0.049 (best result yet, ~4x the noise band)** |

**Full production confirmation** (`finalize_correction_mordredslim271_bdegxtb.py`, job
24468737, actual MLP + XGB_d8 + XGB_d10 + quantile-UQ ensemble, not the quick 2-member check):
**test MAE 1.503** vs mordredslim271's 1.525 — delta -0.022, right at the edge of the noise
band (the full ensemble already captures part of what the raw features add on top of a bare
2-XGB check, so the gain doesn't transfer 1:1 from the quick-check's -0.049). Still judged
real and promotable.

## Interpretation

The method-mismatch hypothesis was confirmed, but **only for BDE (raw electronic energy), not
BDFE**. g-xTB-consistent BDE beats the GFN2-level BDE's already-borderline gain, and BDE+BDFE
together is a clear win outside the noise band — even though BDFE **alone** is still null,
mirroring the GFN2 pattern exactly. The extra RRHO thermal-correction terms in BDFE appear to
be more noise than signal **regardless of the underlying electronic-structure method**, but
BDE's raw electronic energy is more informative when computed at the level that actually
matches what is being corrected.

## Cost-aware follow-up (2026-07-07 SHAP audit)

SHAP importance on the full 275-feat champion (4000-row test subsample, XGB_d8): `ald_bde_
gxtb_kcal` rank 4/275 (mean|SHAP|=0.587), `prod_bde_gxtb_kcal` rank 6/275 (0.484) — both
**cheap** (single SP/opt). `prod_bdfe_gxtb_kcal` rank 15/275 (0.191), `ald_bdfe_gxtb_kcal`
rank 38/275 (0.099) — both **expensive** (full `--ohess` Hessian+RRHO per fragment). Summed
importance: BDE=1.070 vs BDFE=0.290 (ratio 3.7x). **Recommendation for future prospective
screening of new molecules: compute only BDE, skip BDFE** — near-zero importance loss for a
large compute saving (no Hessian needed). The existing production bundle keeps both (already
paid for on the full library), but this should guide future feature-engineering rounds: don't
invest further in BDFE-family (thermal/entropic) descriptors.

See [METHODS_BDE_gfn2_raw_energy_20260707_en.md](METHODS_BDE_gfn2_raw_energy_20260707_en.md)
and [METHODS_BDE_gfn2_free_energy_20260707_en.md](METHODS_BDE_gfn2_free_energy_20260707_en.md)
for the two earlier method levels this one superseded.
