# Homo-Benzoin ΔG Prediction — Project Context & Diagnosis

> Purpose of this file: a self-contained context/handoff doc so any Claude session
> (web, or the VS Code extension) can continue this work without re-explaining.
> Drop it in the repo (e.g. `docs/` or `.claude/context.md`) and point Claude at it.

---

## 1. What the workflow does

- **Task**: predict ΔG for the **homo-benzoin reaction** of aldehydes.
- **Inputs**: aldehyde SMILES + descriptors, plus **product** descriptors.
- **Target / labels**: **DFT-level ΔG at r²SCAN-3c** (dispersion: D4).
- **Baseline / delta-learning source**: **g-xTB single-point** ΔG.
  - Model learns the correction Δ = ΔG(r²SCAN-3c) − ΔG(baseline).
- **Geometries**: optimized at **GFN2-xTB**.
- **Thermal corrections**: **GFN2-xTB** (RRHO). Quasi-harmonic considered, not yet standard.
- **Model**: **XGBoost**. GNN tried, underperforms XGB (expected at this data scale
  with physics-based tabular descriptors).
- **Library scale**: full library of **~220k aldehydes** simulated.

## 2. Current performance

- **MAE ≈ 1.5 kcal/mol**, **R² ≈ 0.85** overall.
- **Key bottleneck**: molecules containing **sulfone (S(=O)₂, S(VI))** or
  **phosphine/phosphorus (P)** groups have large errors.

## 3. Descriptors already tried

xTB, RDKit, Morfeus, Mordred, BDE/BDFE. g-xTB used as single-point energy source.
**SHAP analysis: `P_int` (Morfeus dispersion descriptor) is the single most important feature.**

---

## 4. Core diagnosis (the central hypothesis)

**"P_int is the #1 SHAP feature" AND "sulfone/phosphine is the worst subset" are almost
certainly the SAME phenomenon, not two independent problems.**

Mechanism chain:
1. Morfeus `P_int` integrates the **dispersion potential over the SASA**.
2. Its dispersion coefficients come from **dftd4 (D4)**.
3. D4 atomic polarizabilities are interpolated from **EEQ charge + coordination number (CN)**,
   with reference data fit to **normal-valence** TD-DFT polarizabilities.
4. **Hypervalent sulfur** (S(VI): CN=4, high formal charge, two S=O) and P(III)/P(V) sit at
   the **edge of the D4 reference manifold** → their polarizabilities are **extrapolated / unreliable**.
5. Therefore the model's most-relied-upon feature is **systematically distorted exactly on the
   hardest subset**. Error concentrating on S/P is then mechanistically expected, not coincidental.

### Secondary issue: geometry
- Geometries are **GFN2-optimized**. GFN2 describes **S=O bond lengths and P pyramidalization
  poorly**; the g-xTB single point inherits this bad geometry.
- Suspicion: a fraction of "model error" on S/P is actually **geometry / tautomer / protonation-state
  problems in disguise**.

### Dispersion consistency: LIKELY OK (revised)
- Labels are r²SCAN-3c (**D4**); g-xTB also carries **D4** → dispersion is **same-source on both ends**.
- So the residual is probably NOT driven by "two different dispersion treatments."
- This shifts blame away from "baseline vs reference dispersion mismatch" and **toward the
  molecule-level `P_int` feature itself** carrying the bad D4 polarizability of the sulfone S
  into a whole-molecule integral.

---

## 5. Action plan (ordered by ROI)

### A. Replace molecule-level P_int with carbonyl-LOCAL P_int  ← highest-value, user agreed to explore
- Reaction center = **carbonyl carbon** (and the newly formed C–C / C–OH site).
- Sulfone S is several bonds away from the carbonyl C.
- `atom_p_int` (Morfeus `Dispersion` class, per-atom) integrated over the carbonyl-C neighborhood
  **excludes the bad D4 polarizability of the distant sulfone S**, whereas molecular P_int ingests it.
- Expected effect: turns P_int from "most important but toxic on S/P" into "most important and clean."
- This mirrors cheminit's design: it uses **atom-local** `atom_p_int_Cs`, `atom_p_int_times_area_Cs`
  at the reaction site, never a molecule-level P_int.

### B. Expose dftd4 per-atom intermediates as explicit features
- Per-atom **C6, static polarizability, CN, EEQ charge**.
- Lets the tree branch on "this S has an extrapolated/extreme polarizability" instead of being
  forced to distort P_int to fit.

### C. Geometry QC on high-error S/P
- Re-optimize the S/P subset at **r²SCAN-3c** (pipeline already exists) instead of feeding GFN2
  geometry into g-xTB.
- Manually inspect ~10 worst sulfones: S=O bond lengths, dihedrals, tautomer/protonation state.

### D. Add explicit hypervalent flags
- S=O count, P=O count, formal oxidation state, coordination number.
- Gives the tree a clean split feature for the hypervalent subclass.

### E. Subset-specific SHAP
- Global SHAP reflects the whole 220k library, not the error source.
- **Recompute SHAP on the S/P subset only**, plus P_int **SHAP interaction values**, to confirm
  the model is trying (and failing) to separate the hypervalent subclass.

### F. Thermodynamic decomposition (if熵 is implicated)
- Consider splitting the target into ΔE(electronic) and ΔG_thermal and delta-learning them
  separately. Flexible S/P molecules have many low-freq modes where RRHO entropy is worst.
- Quasi-harmonic / msRRHO (Grimme low-freq damping) on the thermal part.
- Often the R² loss lives entirely in the entropy term.

### G. Uncertainty / applicability domain (productionization)
- Wrap XGB with **conformal prediction** (or quantile / multi-seed ensemble) for per-prediction
  error bars.
- Goal is NOT to make the model accurate on S/P — it's to make the model **know it's uncertain on S/P**
  and auto-route those to DFT (active-learning loop).

### On GNN
- Don't pursue further at this scale. GBTs beating GNNs on hundreds–thousands of physics-tabular
  points is the norm (cheminit itself is LightGBM). A 2D-GNN can't see 3D geometry / xTB electronic
  structure — exactly where S/P physics lives — so it's worst on the hardest subset. Only physically
  meaningful NN direction would be an equivariant/3D model on xTB geometries (MACE-type), which is a
  big project and not worth it while XGB already works.

---

## 6. Reference: the cheminit repo (relevant prior art)

- **jensengroup/cheminit** — "Chemically Intuitive Descriptors for Predicting C–H Borylation
  Regioselectivity" (Jensen group). QM/ML hybrid: RDKit deprotonation → GFN1-xTB CM5 charges
  (smi2gcs) → Morfeus steric+dispersion descriptors → LightGBM pKa sub-model → LightGBM classifier.
- Reproduced successfully end-to-end (2026-07). Key lessons transferred here:
  - **Atom-local, physically-meaningful descriptors at the reaction site** beat whole-molecule
    features and beat GNNs.
  - Morfeus **requires the dftd4 extra** to compute P_int (P_int ← D4). This is the direct link to
    the diagnosis above.
  - Two Morfeus patches needed: `calculators.py` `polarizibilities`→`polarizabilities` typo;
    `sasa.py` add `self.atom_volumes = {atom.index: atom.volume for atom in self._atoms}`.

## 7. Open TODO / next concrete code to write

- [ ] Feature function: extract **per-atom polarizability / C6 / CN / EEQ charge** from xTB/dftd4
      output + **carbonyl-local P_int** (Morfeus Dispersion, per-atom, restricted to reaction-center
      neighborhood).
- [ ] Diagnostic script: **S/P subset residual attribution** + subset-only SHAP + correlation of
      residual with dispersion features + (Δ = DFT − baseline) distribution grouped by S(VI)/P vs rest.
- [ ] Geometry QC harness: flag S/P molecules whose GFN2 vs r²SCAN-3c geometries differ beyond a
      threshold (S=O length, pyramidalization angle).

---
*Last updated: 2026-07-09. Maintainer context: computational chemistry, NHC catalysis / benzoin,
r²SCAN-3c + ORCA pipeline, SLURM/HPC (Groningen).*
