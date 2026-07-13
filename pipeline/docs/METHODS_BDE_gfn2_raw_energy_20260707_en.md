# BDE/BDFE Method Level 1: GFN2-xTB raw electronic-energy BDE

**Status: promoted (borderline-real, later superseded by the g-xTB version — see
[METHODS_BDE_gxtb_20260707_en.md](METHODS_BDE_gxtb_20260707_en.md))**

## Motivation

Bond dissociation energy (BDE) of the aldehyde C(=O)-H bond is the bond mechanistically
activated during Breslow-intermediate formation in the benzoin condensation; the product's
new C-C bond BDE is the bond literally formed in the reaction. The existing 72-feature champion
set only had `wbo_CO` (a Wiberg bond order) as an indirect bond-strength proxy — no real
dissociation energy. This is the cheapest, simplest member of the BDE/BDFE descriptor family.

## Method

`pipeline/compute/calc_bde.py`. For each molecule:
1. Generate the relevant homolysis fragments (aldehyde C(=O)-H radical + H atom; product
   new-C-C radical pair).
2. Run a plain xtb `--opt`/single-point (GFN2-xTB, gas-phase) on parent and each fragment.
3. `BDE = E_el(fragA) + E_el(fragB) - E_el(parent)` — raw electronic energy only, **no
   thermal/ZPE/entropy correction** (that is the Level-2 BDFE variant).

No new software dependency; reuses the project's standard xtb call pattern.

## Coverage

Full library: aldehyde C-H BDE 220,524/220,524 (100%, job 24394017); product new-C-C BDE
219,021/219,022 (99.8%, job 24394018 + retries — 2 chunks hit a deterministic
xtb/`rdDetermineBonds` hang even at 4h timeout, accepted as a ~0.18% known gap). Sidecars:
`aldehydes_bde_descriptors.csv`, `products_bde_descriptors.csv`.

## Result

`finalize_correction_bde.py` (job 24415115), quick 2-member XGB check on top of the
mordredslim271 baseline (271 feats, MAE 1.612):

| config | n_feat | test MAE | delta |
|---|---|---|---|
| mordredslim271 baseline | 271 | 1.612 | — |
| + raw-E BDE (both sides: `bde_prod_CC_kcal`, `bde_ald_CH_kcal`) | 273 | 1.588 | **-0.024** |

Delta sits right at the edge of the established noise band (0.02-0.03 kcal/mol, see
`REPORT_robustness_baseline72_20260702.md`) — judged **borderline-real, small** (same
territory as the earlier "RDKit-no-glob" result). Notably parameter-efficient: only 2 extra
features carry this signal.

## Why this was superseded, not simply kept

The GFN2 level is not the level actually being corrected (the production baseline is g-xTB).
A later hypothesis ("method-mismatch": does computing BDE at the same level as the thing
being corrected carry more signal?) motivated re-deriving BDE at the g-xTB level, which turned
out to give a materially bigger gain (-0.032 vs this level's -0.024) — see
[METHODS_BDE_gxtb_20260707_en.md](METHODS_BDE_gxtb_20260707_en.md). The GFN2-level raw BDE
was never itself promoted to the production bundle; it was a stepping-stone result that
validated "raw electronic energy BDE carries real signal" before the better g-xTB version
replaced it.
