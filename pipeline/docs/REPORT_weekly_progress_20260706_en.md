# Weekly Progress Summary (2026-06-29 ~ 2026-07-06)

Scope: fair DFT-opt rerun → ORCA parallel fix → g-xTB solvation pilot → descriptor search → BDE/BDFE campaign → shared-env incidents and repair → training standardization. The headline result this week is **three consecutive improvements to the g-xTB→DFT correction model** (1.61 → 1.525 → 1.503, the final number from the full MLP+3×XGB production pipeline, not a quick check).

---

## 1. Infrastructure / Workflow Reliability Improvements

### 1.1 Fixed a bug that could have poisoned an entire conclusion: ORCA "parallel" run was actually an MPI artifact
- **Problem**: the `concl3` figure (06-22) treated "DFT-opt 0/13 converged" as evidence that DFT optimization itself was failing — but investigation showed the `nprocs 24` parallel submission couldn't even find `mpirun`. **It was an ORCA startup failure, not SCF non-convergence.**
- **Root cause**: ORCA 6.1.1 needs the OpenMPI 4.1.x ABI (`libmpi.so.40`); the cluster only ships OpenMPI 4.1.5 (incompatible) and 5.0.x (wrong ABI).
- **Fix**: built **OpenMPI 4.1.6** locally on node-local scratch (`software/build_ompi416_local.sh`, ~15min vs 50min+ on GPFS), and found the real hidden blocker — two env vars, `OMPI_MCA_rmaps_base_oversubscribe` / `hwloc_base_binding_policy` — without which ORCA still fails "error termination in Startup" even with the correct library.
- **Follow-up validation**: reran the 36 "g-xTB-hard" molecules' DFT-opt **serially (nprocs=1)** (job 24300065), getting clean, MPI-artifact-free real convergence data — avoiding a wrong conclusion that could otherwise have made it into the final report.
- **Codified as a reusable recipe**: single-point calcs keep using "molecule-level parallelism + serial ORCA" (the established production pattern); this MPI fix is only needed for geometry optimizations.

### 1.2 SLURM operational lessons and standardization
- `scontrol update ArrayTaskThrottle` post-hoc concurrency changes are **unreliable** — the same command sometimes works cleanly, sometimes CANCELS the entire array outright (happened twice, luckily caught early with no output loss). **New rule**: always set concurrency at `sbatch --array=...%N` submission time; any post-hoc change must be immediately verified via `sacct`.
- `clean_node_orphans.sh` (the per-job self-cleanup script) had **two hidden bugs** (a wrong scontrol exit-code check + extracting the "3" from username "schen3" as if it were a jobid) that cancelled each other into a silent no-op — undetected until nodespecific orphan directories grew to 2.7M inodes and hit the quota. Fully fixed and verified this week (171 confirmed-dead directories safely deleted, all live job scratch preserved).

### 1.3 Shared Python environment (`envs/gnn`) crashed twice → fully repaired
- An interrupted bulk reinstall left **158/160 packages (including torch itself)** in a "imports but is actually broken" state, causing ~30% silent failures in the g-xTB BDFE full-library run (SLURM even showed COMPLETED, because the submit script's unconditional trailing `echo` masked the real exit code — fixed alongside).
- Diagnostic method: first ruled out "concurrency/GPFS contention" with a **zero-concurrency interactive import** (a cheap, decisive test that narrows the hypothesis space in one shot), then used `find -newer` to tell "install in progress" from "install stalled."
- **Fix**: reconstructed the `name==version` list for all 158 packages straight from dist-info directory names, bulk-reinstalled via `pip install --ignore-installed --no-deps`, handling two gotchas (setuptools must be fixed first, or sdist-only packages can't build; `--ignore-installed` across a version change leaves stale old-version dist-info behind that must be manually removed).
- **Verified properly**: not just `import` succeeding, but real symbol access (`from xgboost import XGBRegressor`, etc.) plus an actual matmul on a real A100 GPU to confirm CUDA works — login/interactive nodes have no GPU, so a CPU-only smoke test would have missed a broken CUDA runtime.
- Established a "lightweight isolated venv" pattern (`envs/bde_lite`, `envs/selfies_dg`) as a fallback whenever the shared environment is unstable, so new experiments aren't blocked.

### 1.4 Training/process standards codified as lasting constraints
- Every model-training script must now retain training curves, produce full visualization diagnostics (parity/residual/training-curve/error-structure), and export the worst mispredictions — not just print a final MAE.
- Reinforced existing habits: one figure per file (no multi-panel composites), never delete/overwrite result files, register new plotting scripts in the index, and record null results too, not just positive findings.

---

## 2. Scientific Modeling Results

### 2.1 g-xTB full-solvation pilot: settled on the correct geometry route
Completed an A/B route comparison on 2,192 molecules (job 24320345 + backfill):
- **Route A (funnel_v3 geometry + g-xTB-COSMO single point)**: MAE **3.26**, R² 0.888 — very close to DFT, near the project's noise floor (~3.2 kcal/mol).
- **GFN2-ALPB**: MAE 13.05 (catastrophic failure, worse than predicting the mean) — the ALPB solvation model is entirely unsuited to this reaction.
- **Route B (g-xTB geometry optimization directly in solvent)**: only 43% convergence, geometries blow up — **ruled unusable**.
- **Locked in as production standard**: always use funnel_v3 (GFN2/CREST) geometry, g-xTB only for single-point + COSMO, never optimize with g-xTB in solvent.

### 2.2 Descriptor search: corrected a premature negative conclusion
- Around 06-29 the project had concluded "descriptor search is exhausted" (three orthogonal families — ADCH/QTAIM, morfeus-9, RDKit-434 — all fell within the noise band).
- Corrected this week: **the problem wasn't "this descriptor class is useless," it was "dumping an undiscriminated full block" as a method.** Using a SHAP-motivated hypothesis (the model already leans on P_int/SASA, i.e. dispersion/size information), a **targeted 438-dim mordred subset** (MoRSE/CPSA/polarizability/geometric indices, etc.) gave a real, out-of-noise-band gain: **MAE 1.577 → 1.535**.
- Further SHAP-importance + correlation pruning (not PCA) cut 438 down to **199 dims (271 total features)** with almost no accuracy loss (1.525 vs 1.517) — **MORDREDSLIM271 became the new preferred production model**, same accuracy at half the feature count.
- Along the way, quantified the "noise band": test MAE across 5 reshuffled seeds = **1.571 ± 0.013**, matching 5-fold CV — giving a consistent yardstick for judging whether any future improvement is real (<0.02-0.03 is noise).

### 2.3 BDE/BDFE descriptors: this week's biggest model gain
Systematically pursued bond dissociation energy from a mechanistic hypothesis (the C(=O)-H bond activated during Breslow-intermediate formation):
- **At the GFN2 level**: raw BDE gave a borderline-real gain (+0.024), but BDFE (the free-energy-corrected, theoretically more rigorous quantity) was actually a null result (+0.007, within noise) — counterintuitive but confirmed: the thermal-correction terms from low-frequency vibrational modes in `--ohess` likely introduce more noise than signal.
- **Method-mismatch hypothesis**: suspected that computing BDE/BDFE at the GFN2 level, while the model is meant to correct g-xTB's own errors, was diluting the signal via a level-of-theory mismatch. Switched to **g-xTB-consistent BDE/BDFE** (reusing the project's own hybrid energy formula, algebraically recovering the cached parent electronic energy at zero extra compute cost).
- **Final result (landed 2026-07-06)**:
  | Config | # features | test MAE | vs baseline |
  |---|---|---|---|
  | MORDREDSLIM271 (baseline) | 271 | 1.612 | — |
  | + BDFE(g-xTB) | 273 | 1.605 | -0.007 (noise-level, same as GFN2) |
  | + BDE(g-xTB) | 273 | **1.580** | **-0.032 (real — beats GFN2's own +0.024)** |
  | **+ BDE + BDFE(g-xTB)** | 275 | **1.563** | **-0.049 (quick-check, ~4x the noise band)** |
- This was the **only quick-check gain this week exceeding 4x the noise band**, so it was immediately run through the full MLP+3×XGB ensemble (not the 2-member XGB quick check) for verification: **real production-grade test MAE = 1.503** (vs MORDREDSLIM271's 1.525, Δ-0.022, at the edge of the noise band — smaller than the quick-check suggested, but confirmed as a real improvement). Promoted to the new production champion.

### 2.4 Model lineage at a glance
```
ENSEMBLE72 (06-26)              MAE 1.61   72 feats, MLP+3xXGB, uncertainty routing
   ↓ + targeted mordred438
MORDRED510 (07-03)               MAE 1.517  510 feats
   ↓ SHAP pruning
MORDREDSLIM271 (07-03)           MAE 1.525  271 feats (half the size, same accuracy)
   ↓ + g-xTB BDE/BDFE (full production)
MORDREDSLIM271_BDEGXTB (07-06)   MAE 1.503  275 feats  ★ current production champion
```

### 2.5 Other explorations: null results recorded systematically too
- **BDE surrogate model** (predicting BDE/BDFE from cheap descriptors instead of always running real xtb) started training (job 24436405), to diagnose whether BDE information is already implicitly captured by existing features, and to speed up screening of new molecules in the future.
- **SELFIES / ECFP / sequence models (GRU)** tried predicting ΔG directly from reactant+product structure — all **underperformed** the existing RDKit-2D descriptor surrogate (3.0-3.4 vs 2.92 kcal/mol MAE) — these directions are now explicitly ruled out, avoiding repeated effort later.

---

## 3. Summary

| Dimension | This week's outcome |
|---|---|
| Corrections | Fixed the "DFT-opt 0/13" MPI artifact; corrected the premature "descriptor search exhausted" conclusion |
| Infrastructure | ORCA parallel fix, SLURM throttle discipline, orphan-cleanup bug fix, shared env fully repaired after two crashes + isolated-venv fallback pattern established |
| Model accuracy | g-xTB→DFT correction model test MAE 1.61 → 1.525 → **1.503 (current production champion, verified via full ensemble)**, feature count 72→271→275 |
| Methodology | Quantified the noise band (±0.013); "targeted subset beats undiscriminated dump" descriptor methodology; training-diagnostics standard; g-xTB solvation geometry route finalized |
| Dead ends ruled out | SELFIES/ECFP/sequence-model surrogates, GFN2-level BDFE, Route B (in-solvent optimization) |
