# CREST conformer search vs the RDKit funnel (2026-06-14)

## Why
The benzoin ΔG labels are limited by conformer-SEARCH stochasticity, not ensemble
size (`conformer-search-noise`). The v2 robust funnel (deterministic embed + RMSD
prune + denser GFN-FF prescreen) reduced scatter but `diag_conf_connectivity.py`
exposed a new failure: the denser GFN-FF prescreen can relax a conformer into a
**broken-connectivity** structure (bond formed/broken or fragmented) whose
spuriously low xTB energy then dominates the Boltzmann average and poisons the DFT
label. RDKit ETKDG + a plain GFN-FF `--opt` has no topology guard.

CREST (Grimme) does metadynamics-based sampling with an internal topology check that
discards connectivity-changed structures, so its ensemble is — by construction —
genuine conformers of the input. That removes exactly this failure mode.

## What was built
- **CREST 3.0.2** installed at `/home/schen3/crest/crest/crest` (static binary).
- **`pipeline/compute/conf_crest.py`** — `rank_conformers_crest(...)`, same interface
  as `conf_funnel.rank_conformers_funnel` (returns energy-sorted `[(xyz, E_Eh)]`), so
  it is a drop-in for the featurizer's downstream top-K Boltzmann + r2SCAN-3c ΔG.
  - `method="gfnff"` / `"gfn2"` — native CREST search levels.
  - `method="gfn2//gfnff"` (default) — composite done **manually**: CREST `--gfnff`
    metadynamics sampling, then `thermo_orca._xtb_opt_energy` (GFN2 `--opt tight`) on
    the top-L members. Returned energies are GFN2 — apples-to-apples with the funnel,
    and ~60× cheaper on the GFN2 stage than CREST's own multilevel.

### Why the composite is manual, not native (CREST 3.0.2 GNU build)
The native composite does NOT work in this static binary:
- CLI `-gfn2//gfnff` errors: *"not yet available with new calculator"* (tblite).
- TOML multilevel (two `[[calculation.level]]` blocks gfnff+gfn2, `[dynamics]
  active=[1]`, `runtype=imtd-gc`) parses and runs the GFN-FF MTD fine, but then
  **SEGFAULTS** in the "Multilevel Ensemble Optimization" while optimizing all ~600–700
  trajectory snapshots at GFN2 (*"BLAS: Bad memory unallocation"*, SIGSEGV across
  threads). Reproducible across `multilevelopt` true/false, `optlev` tight/normal, and
  `ulimit -s unlimited` + `OMP_STACKSIZE=4G`. It is a threaded-GFN2/BLAS bug in the
  prebuilt GNU binary. Note also `ewin` is NOT a top-level TOML key (pass `--ewin` on
  the CLI; CLI args override the TOML). The Intel CREST build may avoid the BLAS bug.
- The manual two-leg is also the better design: it GFN2-optimises only the top-L
  conformers of CREST's already-deduplicated, topology-checked ensemble (~10 opts),
  not all ~700 — far cheaper, and stable.
  - Belt-and-suspenders topology guard (`rdDetermineBonds` vs input graph) on top of
    CREST's own check.
- **`pipeline/sample_conftest_v2.py`** → `data/library/subset_conftest_v2.csv` — a
  CLEAN conformer-test set replacing the dirty `subset_conftest.csv` (whose first row
  was an SF5 reactive_group, several rows out of the aromatic-only scope). **80
  molecules** (`--per-cell 10`): in-scope (aromatic carbo+hetero, `classify()=='keep'`),
  balanced 2 classes × 4 benzoin-PRODUCT rotatable-bond bins (rigid 0-3 / 4-7 / floppy
  8-12 / very 13+), MaxMin-diverse. Product rotbonds span 3-27. (Bumped 40→80 for a
  meaningful catastrophic-failure RATE, which is only ~4%.)

## Results (problem molecules from the diag)
| molecule | method | n | broken | lowest broken? | time (8 cores) |
|---|---|---|---|---|---|
| fused furan-dione benzoin (39 bonds) | funnel_v2 | 20 | 0 | False | 140 s |
| | **crest gfnff** | 20 | **0** | **False** | 464 s |
| acetoxy-pyrrole benzoin (37 bonds) | funnel_v2 | 20 | 0 | False | 215 s |
| | **crest gfnff** | 20 | **0** | **False** | 682 s |

- CREST GFN-FF returns **clean ensembles (0 broken, lowest conformer correct)** on
  both molecules — the broken-connectivity poisoning is gone.
- The composite `gfn2//gfnff` validated end-to-end on the benzaldehyde benzoin: 10
  conformers, proper GFN2 energies (~-44.21 Eh), all topology-clean, 173 s.
- Cost: CREST GFN-FF search is **8-11 min/mol** on a floppy benzoin (vs ~2-4 min for
  the funnel) — the robustness tax. Acceptable for labelling; budget accordingly.

## A/B comparison — LAUNCHED 2026-06-14
`runs/launch_crest_comparison.sh` submitted two featurize arrays over the 80-mol
`subset_conftest_v2` at r2SCAN-3c, K=3 — same molecules/DFT, only the conformer search
differs:
- **funnel_v2** (job 23812080) → `data/raw/featurize_conftest2_funnelv2`
- **crest** composite gfn2//gfnff (job 23812081, `featurize_crest.py`, CREST_CORES=8)
  → `data/raw/featurize_conftest2_crest`

Single-molecule pre-flight (rigid, xTB-only) passed: dG_xtb=-10.06, ~7 min.
Analyse with `pipeline/compare_conf_methods.py` (runs incrementally; flags |Δ dG_orca|
> 2 kcal as search-driven disagreements). Expected win: methods agree on most, and the
flagged few are funnel broken-topology failures (verify with diag-style topology checks
on the flagged subset).

## A/B RESULT + attribution (2026-06-14, n=59 of 80)
Δ(dG_orca) = crest − funnel_v2: mean **+0.99**, std 4.56, **MAD 3.17 kcal**, max|Δ|
15.98; **30/59 differ >2 kcal, 11 >5, 4 >10**. So the conformer SEARCH is a first-order
label-noise source (not a side effect). Attribution by topology of each method's lowest
benzoin conformer (`attribute_disagreements.py`) splits the big disagreements into two
regimes:
- **funnel broken-topology failures (CREST right).** idx 288552: funnel's lowest was
  BROKEN (7/20 broken), 38 kcal spuriously below the lowest intact → ΔG −10.5 (artifact);
  CREST clean → +1.6. These are the catastrophic label poisons.
- **genuine conformer divergence (no broken topology).** idx 176773, 11280 (large fused
  polycyclics, 53/61 bonds): both lowests intact but different basins; the funnel's dense
  ETKDG actually found a 3.8–4.2 kcal LOWER GFN2 minimum than CREST's GFN-FF MTD. CREST
  under-samples here.

**Conclusion: don't switch engines — guard the funnel.** The funnel's *only* catastrophic
failure is a broken-topology lowest conformer; CREST's robustness comes from a topology
check, but its MTD under-samples big molecules where the funnel wins. So
`conf_funnel_v3 = conf_funnel_v2 + topology guard` (drop conformers whose perceived
heavy-atom graph ≠ input, before Boltzmann) keeps the dense sampling AND gains the
robustness, for one RDKit perception per conformer.

**funnel_v3 validated on the catastrophe (idx 288552):** drops the 7 broken (20→13),
lowest now intact at −87.688 Eh — removing the 38.2 kcal spurious stabilization and
landing within ~1 kcal of CREST's −87.687. So v3 recovers CREST's correct answer at ~zero
cost.

## CREST's lasting role
CREST was the instrument that exposed the failure mode and the independent reference that
validated the cheap fix. Keep `conf_crest.py` as a robust cross-check / second opinion on
suspicious labels, not as the production engine.

## Three-way A/B — LAUNCHED
- funnel_v2 (job 23812080) → `featurize_conftest2_funnelv2`  (broken-prone baseline)
- crest      (job 23812081) → `featurize_conftest2_crest`     (robust ref, under-samples big)
- **funnel_v3** (job 23816692) → `featurize_conftest2_funnelv3` (guard + dense — the candidate)
Expected: v3 ≈ crest on the broken cases (both correct) and v3 ≈ v2 elsewhere, but with
the catastrophic >10 kcal disagreements removed. Confirm with `compare_conf_methods.py
--a …funnelv3 --b …crest` and `--a …funnelv2 --b …funnelv3`.

## Three-way RESULT (dG_orca MAD, kcal/mol)
| comparison | n | MAD | >2 kcal | >10 kcal | reading |
|---|---|---|---|---|---|
| **v2 → v3 (the guard)** | 43 | **0.46** | 2 | 1 | surgical: ~95% labels identical, only the broken cases move (one by +13) |
| v2 → crest (engine swap) | 65 | 3.27 | 36 | 4 | swapping engines perturbs *everything*, incl. under-sampling big mols |
| v3 → crest | 42 | 2.33 | 19 | 1 | guard moved funnel toward the robust ref; residual = genuine divergence |

The v2→v3 MAD of 0.46 kcal is the headline: the topology guard is a SURGICAL fix — it
leaves the well-behaved ~95% of labels untouched and only rewrites the ~2-5%
broken-topology catastrophes. A full CREST swap (MAD 3.27) changes every label and loses
sampling on large polycyclics. v3 is strictly closer to CREST than v2 (3.27→2.33, >10 kcal
cases 4→1) — it adopts CREST's robustness exactly where CREST is right, and keeps the
funnel's denser sampling everywhere else.

## Broken-rate + final confirmation (n=74/80)
v2→v3 over 74 molecules: MAD **0.31 kcal**, and the guard fired on **exactly 2 molecules
(~2.7%)** — both genuine broken-topology catastrophes — leaving the other 72 within ~0.1
kcal (numerical noise). This matches the earlier ~4% catastrophic-failure estimate.
| idx | v2 (broken) | v3 (guarded) | crest (ref) | note |
|---|---|---|---|---|
| 288552 | −10.52 | +2.68 | +1.58 | v3≈crest (~1 kcal): broken low removed |
| 241362 | +4.91 | −0.06 | +2.94 | both un-poisoned; v3/crest pick different *intact* minima (v3 lower-E) |
So the guard is surgical AND correct: where it fires, v3 lands on the funnel's lowest
INTACT conformer and agrees with CREST that v2's value was an artifact.

## ⚠️ CORRECTION (during the re-label): production labels were UNDERSAMPLED, not v2
The 3% broken-rate above is funnel_v2-vs-v3 (both DENSE search). But the PRODUCTION
labels in `data/featurize.parquet` were made with the **legacy `_rank_conformers`**, whose
`_auto_nconfs` returns **1 conformer for ≤3 rotatable bonds** (and only modestly more
otherwise) — drastically undersampled. So the funnel_v3 re-label is a **legacy→dense
overhaul**, not a guard tweak. Restricted to the 1627 in-scope TRAINING molecules
(first 256 re-labeled): Δ(v3−old) **MAD 3.86 kcal**, mean +0.95, **76% change >1 kcal,
24% >5, 7% >10**, max 33. The old parquet even holds unphysical labels (−83, +50 kcal) =
single-conformer search failures (QC mag_max=45 drops the worst).

**Implication (positive):** the ~2.68–3.2 kcal model "noise floor" was largely
CONFORMER-SEARCH-noise-limited (sparse legacy sampling), NOT a fundamental xTB↔DFT method
gap. Properly sampled funnel_v3 labels should lower it. This also fully justifies
re-labeling all 3063 (it's a quality overhaul of the entire training set, not a 3% patch).
Relates to [[conformer-search-noise]], [[delta-mae-noise-floor]].

## Interim retrain (partial, n=362 matched in-scope, ~641/3063 re-labeled)
Same molecules, OLD vs NEW labels, Δ-model (xgb) CV:
| labels | MAE | RMSE | max |
|---|---|---|---|
| old (legacy sparse) | 2.645 | 3.507 | 11.72 |
| **new (funnel_v3)** | **2.209** | **2.911** | **8.68** |
MAE −0.44 (~16%), **RMSE −0.60 > MAE drop** → the gain is in the TAIL (conformer
catastrophes), as predicted. The ~2.68 noise floor WAS largely conformer-search noise.
Caveats: partial non-random subset (parquet order); CV measures label self-consistency,
but funnel_v3 labels are independently validated better, so the drop is genuine.

## Full zoo on clean labels (partial, n=473 in-scope) — `retrain_report.py`
Δ-model zoo DFT-level MAE: ridge **2.201**, xgb 2.226, gpr 2.237, gbt 2.251, rf 2.257,
stack 2.264, hist_gbm 2.285 — all ~2.2-2.3. 2D surrogate (pure SMILES, no xTB): xgb 3.19.
AD by BENZOIN-PRODUCT flexibility: rigid 1.78 → mid 1.94 → floppy 2.22 → very_13+ 2.79;
by category carbo 2.30 ≈ hetero 2.27.

**Refined conclusion:** the noise floor dropped **2.68 → ~2.20**, and **linear ridge now
TIES xgb/gpr** — removing the conformer noise made the descriptor→ΔG map more linear, so
the residual is near irreducible label noise. Ridge is a defensible (simpler) production
choice on clean labels. Tooling caveat fixed: `delta_core.build_model` only knows
xgb/rf (else → GradientBoosting), so the real ridge/GPR come from
`explore_models.regression_zoo`, not `build_model`.

## Known limitation (found during re-label)
funnel_v3 embeds a fixed 400 conformers for 8–12 rotbonds regardless of molecule SIZE,
so a very large benzoin product (e.g. idx 5798: a 66-heavy-atom dimer of a polycyclic
epoxide-lactone aldehyde) makes the GFN-FF prescreen grind ~8h through per-xtb 5-min
timeouts on non-converging conformers (task 23818958_47, cancelled). The SLURM
`--time=10:00:00` cap auto-kills such stragglers (they just drop out — 1–2 negligible
molecules), but BEFORE production rollout add a size-aware conformer cap (scale n_confs
down for >~55 heavy atoms) and/or an overall per-molecule wall-clock to funnel_v3.

## FINAL RESULT (2026-06-15, full clean-label set n≈1598 in-scope, 3049/3063 re-labeled)
`retrain_report.py --new data/featurize_funnelv3.parquet`:
- **Matched old-vs-new (n=1570, same molecules):** Δ-model MAE **2.691 → 2.247**, RMSE
  3.60 → 3.04, max 13.4 → 11.7.
- **Δ-model zoo (clean labels):** gpr **2.185** · xgb 2.203 · ridge 2.216 · hist_gbm/gbt
  2.217 · bayes_ridge 2.225 — top models tie ~2.19-2.22, R²≈0.59. (mlp/knn/kernel_ridge
  worse.) **Noise floor 2.68 → ~2.19, −18%.**
- **2D surrogate (pure SMILES, no xTB):** xgb **2.924** · gpr 3.165 · ridge 3.177
  (was ~4.0 on old labels). Hierarchy gap to the Δ-model ≈ 0.73 kcal.
- **AD (new Δ-model):** category carbo 2.171 < hetero 2.342; product-flexibility
  rigid 1.725 ≈ mid 1.748 → floppy 2.240 → very_13+ 2.765 (the AD axis).

**Takeaway:** the conformer-search overhaul (born from the CREST investigation) is the
single biggest model-quality lever found — it broke the 2.68 floor that earlier
diagnostics had called fundamental. Both hierarchy tiers improved; ridge≈xgb≈gpr confirms
the residual is near irreducible label noise.

## Ship (after 100%)
1. Re-assemble at 100%; back up old `data/featurize.parquet` → `*_legacy_sparse.parquet`;
   make funnel_v3 the production table.
2. `train_delta.py`/`sweep_delta.py` → `assemble_model.py` → `src/benzoin_dG/models/`
   + per-category `category_ad.json`. Ship xgb (pipeline-standard; gpr available for σ).
3. Apply the funnel_v3 size-aware conformer cap (see Known limitation) for production
   inference. CREST stays a cross-check.
2. Promote **funnel_v3** to the production featurize search (replace v2) and re-label the
   3063-mol parquet — cheap (funnel speed + a perception). Submit to **genoa** `%192`.
3. Re-check the n=1500 ~7 kcal residual after the guard: how much of it was these
   broken-topology poisons vs the genuine xTB↔DFT method gap.
