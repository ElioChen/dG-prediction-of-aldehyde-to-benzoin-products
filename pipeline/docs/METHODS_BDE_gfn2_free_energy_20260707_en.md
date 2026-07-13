# BDE/BDFE Method Level 2: GFN2-xTB free-energy-corrected BDFE

**Status: NULL RESULT — not promoted**

## Motivation

Raw electronic-energy BDE (Level 1) ignores zero-point energy, thermal, and entropic
corrections. BDFE (bond dissociation *free* energy) is the thermodynamically rigorous
quantity and was expected to correlate better with the reaction's true energetics —
especially the aldehyde C-H bond (directly tied to Breslow-intermediate formation
thermodynamics) and the product C-C bond (the bond literally formed).

## Method

Two implementations, v1 then v2 (v2 fixes a solvent-consistency bug in v1):

- **v1** (`pipeline/compute/calc_bde_free_energy.py`): xtb `--ohess` (full numerical Hessian +
  RRHO thermal correction) on parent + each radical fragment, **gas-phase** (no implicit
  solvent), plus a Shermo qRRHO re-correction on the g98.out (reusing `thermo_orca.py`'s
  `run_ohess`/`run_shermo`/`parse_shermo_G` pattern). The isolated H-atom fragment uses an
  analytic ideal-monatomic-gas correction (Sackur-Tetrode + electronic doublet degeneracy —
  no vibrational Hessian possible for one atom).
- **v2** (`pipeline/compute/calc_bde_free_energy_v2.py`): same physics, but (a) reuses the
  parent's G/E from the existing funnel_v3 featurization instead of recomputing it (only the
  two new radical fragments need a fresh `--ohess`, roughly halving to two-thirds the xtb
  cost), and (b) adds `--alpb dmso` to all fragment calculations. **v1 was gas-phase
  throughout (parent included) — mixing a solvated-parent number with gas-phase fragments
  (or, in the case of the original parent recompute, being gas-phase everywhere) was a
  methodological inconsistency** relative to the rest of the project's DMSO-solvated
  methodology, not merely a rounding difference. Re-validated: switching to DMSO-consistent
  fragments moved values from a 4.6 kcal/mol gap vs v1 down to a small, physically-expected
  ~1.7 kcal/mol residual (differential solvation stabilization between the closed-shell
  parent and the open-shell radical fragments — a real effect, not a bug).

## Pilot validation

20-molecule pilot (10 aldehyde + 10 product), 100% success: G-corrected BDE runs ~14-19
kcal/mol **lower** than the raw electronic energy (aldehyde C-H: E 102.8-116.7 → G_xtb
89.0-102.3, now matching the ~88 kcal/mol experimental ballpark at the low end; product C-C: E
75-107 → G_xtb 58-90). Shermo vs xtb's own RRHO agreed within 0.5 kcal/mol — no severe
low-frequency-mode pathology in this sample.

A bug was caught and fixed along the way: `_frag_G` initially hardcoded `--uhf 1` for every
species including the closed-shell parent (should be `uhf=0`), causing immediate xtb fatal
errors; fixed by threading a per-species uhf parameter through.

## Coverage

Aldehydes (v2, job 24416982): 1459/1471 (99.2%, 12 timeouts accepted). Products (v2, job
24422675 after a resubmit at 120min/chunk to fix a 28% timeout epidemic at 60min/chunk):
1435/1463 (98.1%, 28 timeouts accepted). Sidecars: `aldehydes_bdfe2_descriptors.csv` (218,724
rows, 97.9% filled), `products_bdfe2_descriptors.csv` (215,367 rows, 96.0% filled).

## Result

`finalize_correction_bdfe_full.py` (job 24434658), same quick-check protocol as Level 1:

| config | n_feat | test MAE | delta |
|---|---|---|---|
| mordredslim271 baseline | 271 | 1.612 | — |
| + BDFE (GFN2, both sides) | 273 | 1.605 | **-0.007 (null)** |

Delta is well below the 0.02-0.03 noise band, and notably weaker than Level 1's raw-E BDE gain
(-0.024). An aldehyde-only quick check (job 24427785) gave the same story in isolation:
+0.003, essentially nothing.

## Interpretation

Despite BDFE being the thermodynamically more rigorous quantity, it did **not** outperform the
cruder raw electronic energy for this ML task. Plausible explanation: the extra thermal/
entropic correction terms from `--ohess` (especially low-frequency vibrational modes, which
are notoriously mode-dependent and numerically noisy) may introduce as much noise as physical
signal, while the raw electronic energy is a cleaner (if less "correct") per-molecule quantity
that happens to correlate better with what the correction model needs.

**Decision: do not promote GFN2-level BDFE.** This result held up again at the g-xTB level
(Level 3) — BDFE alone remained null there too, confirming the pattern is about the RRHO
correction itself, not the underlying electronic-structure method.
