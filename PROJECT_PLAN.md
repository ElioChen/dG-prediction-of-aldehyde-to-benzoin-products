# Benzoin Reaction ΔG Prediction — Project Plan

> **Living document.** Multi-level map of every component of the project, each
> marked ✅ completed · 🔄 in progress · ❌ missing / not started · 🅿️ parked
> (tried, deprioritized). For each component: the **principle** (why it is the
> right step, and the chemistry / statistics behind it), the **status**, and
> **what is still missing**.
>
> Created 2026-09-10 at the user's request. Companion docs:
> `LAB_JOURNAL.md` (reasoned daily narrative of every operation),
> `RUN_LOG_20260903.md` (raw dated log), `PROJECT_SUMMARY_20260907.md`
> (reporting draft), `HANDOFF_<date>.md` (session handoff).
>
> Top-level structure requested by the user:
> **1. dG Simulation Workflow · 2. Prediction Optimization Workflow**
> (3. Catalyst Space is out of scope — see §3; + 4. cross-cutting infrastructure).
>
> Bilingual: English here, 中文 in `PROJECT_PLAN_ZH.md` (keep in sync).

---

## 0. Goal & scientific framing

**The reaction.** Benzoin condensation: two aldehydes (a *donor* R¹CHO and an
*acceptor* R²CHO, possibly the same molecule) couple to an α-hydroxyketone
("benzoin") product R¹C(=O)–CH(OH)R². Catalyzed in practice by an
N-heterocyclic carbene (NHC) or cyanide via an acyl-anion ("Breslow")
equivalent. When donor = acceptor it is the **homo** case (2 A → AA); when
they differ it is the **cross** case (A + B → AB).

**The quantity we predict.** The reaction free energy
**ΔG = G(product) − G(donor) − G(acceptor)** in kcal/mol, in DMSO at 298 K, at
the r2SCAN-3c / CPCM(DMSO) level with a GFN2-xTB RRHO thermal correction.
This is the *thermodynamics* of the coupling — it tells us which pairings are
downhill (favorable products) and by how much. It is **not** the kinetic
barrier and **not** catalyst-dependent; the catalyst side is a separate project
(§3, out of scope here).

**Why predict it instead of computing it.** The useful design question is
"given ~10⁶ possible aldehyde pairs, which ones give a favorable benzoin?"
A DFT ΔG per pair costs hours of CPU; a trained model costs milliseconds.
So the project is a **screening / ranking** problem powered by a surrogate
model, with DFT used only to generate training labels and to arbitrate the
model's most uncertain calls.

**What "done" looks like.**
- (Goal 1) A trustworthy ΔG dataset + simulation workflow that can label any
  new pair on demand.
- (Goal 2) A model whose scaffold-disjoint holdout error is at or near the
  label-noise floor, delivered as a deployable tool (`predict_dg.py`) with
  calibrated uncertainty and a favorable/unfavorable + rank decision layer.
- (Goal 3) Deployment: score large candidate libraries, hand chemists a ranked
  shortlist, and route the model's high-risk calls to DFT.

**Current headline (2026-09-10).** cross **reference model** = r1-10 blend,
scaffold-disjoint holdout MAE **2.215 kcal/mol** (g-xTB baseline 5.037), on the
≈ 2.9 kcal/mol single-conformer label-noise floor. Goal 3 tooling shipped.
**The project is being re-based** on the user's 2026-09-10 input: the cross
chemical space was mis-defined (`candidates_v3` ≠ the real 220,860² pair space),
so cross AL will be redone over a proper **"flying dataset"** (§2.11); the past
AL rounds and the r1-10 champion are kept as reference, not the forward line.
Active compute: (a) **B97-3c cheap-baseline lever** (Tier B relabel, may break
the 2.9 floor) and (b) a **from-scratch full-library homo model** with
recomputed DFT labels. `Catalyst Space` (§3) is **out of scope** for this project — this project is
the substrate axis only.

---

## 1. dG Simulation Workflow — ground-truth ΔG labels

*Principle of the whole workflow: **one method, everything saved.** Every
species (each aldehyde, each product) is treated with the identical conformer
search + geometry + thermal + single-point recipe, and all geometries,
energies and descriptors are persisted and cross-linked by a stable integer
`id`. This makes ΔG a clean difference of consistently-computed free energies
and lets any later step (relabel, feature, audit) reuse the intermediates.*

### 1.1 Reaction & thermodynamic target — ✅
- **Principle.** ΔG of A + B → AB. Free energy, not electronic energy, because
  entropy (losing one molecule of translational/rotational freedom on coupling)
  is a first-order effect (~+10 kcal/mol unfavorable) and varies with substrate
  size. Solvent (DMSO) matters because the product's H-bonding α-hydroxyketone
  is stabilized differently than the two aldehydes.
- **Status.** Definition frozen. `dG = (G_prod − G_don − G_acc) · 627.509`.
- **Missing.** Nothing.

### 1.2 Aldehyde structure library — ✅ (this is the ground truth; see also §2.11)
- **Principle.** The universe of monomers. `data/library/aldehydes_clean_v6.csv`
  = **220,860 aldehydes** — a large enumeration filtered to synthesizable
  mono-aldehydes, categorized (`cho_class` ∈ aliphatic / aromatic-carbocyclic /
  aromatic-heterocyclic) for class-balanced sampling, with an `xtb_risk` flag.
  **The v6 library is deliberately inclusive**: it contains molecules that will
  *fail* GFN2 optimization or DFT (strained, hypervalent, huge, pathological
  conformer surfaces). That attrition (~5–15% through the pipeline) is a
  *property of the library*, not a pipeline bug — a "can't compute this one"
  outcome is a valid, recorded result.
- **Status.** Downstream caches: `homo_v6/aldehydes_all.csv` 209,526 rows with QM
  descriptors; ~220k with Mordred / BDE; homo product library `products_all.csv`
  184,199 rows with a valid built structure (the gap to 220k ≈ the uncomputable
  fraction + build failures).
- **Missing / risk.** The recovered/rebuilt CSVs carry defects (surfaced
  2026-09-10): (a) `"2.0"`-style float ids that string-join to ~0 rows unless
  normalized (`qc.norm_id`); (b) the `G_xtb` column is **id-misaligned for ~8%
  of rows** (thermal −84…+87 Ha, or subtle e.g. 1.23 Ha on a 19-atom aldehyde;
  same values in the home backup → not a purge corruption, a featurize bug).
  **Rule: bounds-check every stored xTB energy / thermal before use** (per-atom
  band 0.001–0.020 Eh/atom for the Gibbs thermal correction).

### 1.3 Conformer generation — `conf_funnel_v3` — ✅
- **Principle.** A molecule's free energy depends on which 3D conformer you
  evaluate; the wrong conformer is the dominant single source of label noise.
  `funnel_v3` is a staged search: RDKit ETKDG ensemble → cheap GFN-FF / GFN2
  screening → keep the low-energy funnel → a topology guard rejects embeds that
  changed bonding. "v3" fixed a bug where a bad embed silently corrupted a label.
- **Status.** Production method for every species since round 7. Superseded
  CREST and earlier funnel variants (documented as slower / no accuracy gain).
- **Missing.** Nothing for the main line. Multi-conformer *Boltzmann-averaged*
  labels were tried as a noise-reduction lever — **2× confirmed null** (§1.7),
  so single lowest-G conformer stays.

### 1.4 Semiempirical geometry + thermal — GFN2-xTB `--ohess tight --alpb dmso` — ✅
- **Principle.** DFT geometry optimization + Hessian on ~200k species is
  unaffordable. GFN2-xTB is a tight-binding method ~1000× cheaper that gives
  geometries good enough for a DFT single point, plus a full numerical Hessian
  → the RRHO (rigid-rotor / harmonic-oscillator) **thermal correction**
  `G_xtb − E_el_xtb` (ZPE + thermal enthalpy − TΔS). We reuse this xTB thermal
  on top of the DFT electronic energy: `G_lvl = E_lvl(DFT SP) + (G_xtb − E_el_xtb)`.
  This is standard "composite" thermochemistry — the thermal correction is
  much less method-sensitive than the electronic energy, so a cheap Hessian is
  acceptable.
- **Status.** Production. Geometries + xTB G/E persisted; a 2026-09 archive
  (`bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst`) holds ~180k of
  them.
- **Missing.** The 2026-09 geom archive is **partial for chunks ~1400–1830**
  (~2.5% of pairs have no usable geometry) → those need a fresh GFN2 opt.
  Bad stored thermal (§1.2) → recompute via a fresh `--ohess` on the archived
  geometry (wired into `homo_sp_from_geom_worker.py` 2026-09-10).

### 1.5 DFT single-point — 🔄
- **Principle.** The label's accuracy tier. Three levels are in play:
  - **r2SCAN-3c / CPCM(DMSO)** — the project **label**. A "3c" composite:
    the r2SCAN meta-GGA functional with a purpose-built triple-ζ basis (def2-mTZVPP),
    D4 dispersion and a geometrical counterpoise correction. Chosen as the best
    accuracy-per-cost for a 200k-scale campaign; benchmarked against higher
    functionals (wB97X-3c parser never worked → parked).
  - **B97-3c / CPCM(DMSO)** — a candidate **Δ-learning baseline** (the "cheap
    baseline lever", Rec-1 / work-set D). A GGA composite, ~5–20× cheaper than
    the r2SCAN-3c label. Pilot (128 pairs, same geometry for all 3 SPs):
    residual `dG_r2scan − dG_b973c` = a near-constant −5.24 kcal offset (which a
    Δ-model absorbs trivially) with **std 1.11**, versus g-xTB's std 4.32.
    Std ratio 0.26 → the Δ-model's theoretical error floor drops ~4× if we swap
    the baseline. This is the first real accuracy lever in months.
  - **g-xTB / COSMO(DMSO)** — the current, cheapest Δ-learning baseline
    (semiempirical). Kept for diagnostics and as a fallback.
- **Status.**
  - r2SCAN-3c labels: ✅ for cross rounds 1-10 (35,528 pairs); ❌ for the full
    homo library (physically lost in the purge — 30k of ~219k survive; no backup
    anywhere).
  - **cross Tier B campaign** 🔄: recompute all 35,528 cross pairs
    self-consistently (fresh geometry + r2SCAN-3c AND B97-3c AND g-xTB on that
    one geometry). ~72% done 2026-09-10, drain ~09-11/12. On drain → retrain the
    champion with `BASELINE_COL = dG_b973c_kcal`.
  - **homo full-library relabel** 🔄: SP-on-archived-geometry (r2SCAN-3c label +
    B97-3c baseline), 179,431 pairs; + a 4,621-pair regen track for the geometry
    gap. Restarted clean 2026-09-10 after three contamination bugs were fixed.
- **Missing.** Tier B merge + champion retrain + verdict (does B97-3c break the
  2.9 floor at full scale?). Homo full-library labels. wB97X-3c / higher-tier
  arbitration is parked, not needed.

### 1.6 Composite free energy & ΔG assembly — ✅
- **Principle.** `G_lvl(species) = E_lvl(SP) + thermal_xtb(species)`;
  `dG_lvl = G_lvl(prod) − G_lvl(don) − G_lvl(acc)` (cross) or
  `G_lvl(prod) − 2·G_lvl(ald)` (homo). Assembled per pair into the training
  table alongside features.
- **Status.** `assemble_cross_training_table*.py`,
  `homo_sp_from_geom_worker.py`. Working.
- **Missing.** A `--full-library` mode of `assemble_homo_standalone_table.py`
  (column availability verified; to be written when homo labels land).

### 1.7 Label-noise characterization — ✅ (settled)
- **Principle.** You cannot train a regressor below the noise in its labels.
  The label here is one r2SCAN-3c energy on one xТБ conformer; the conformer
  choice injects noise.
- **Result (3 independent probes, converged 2026-09-04).**
  1. Conformer-noise probe (32 products × 5 conformers): per-product
     `dG_std` mean **2.975** / median 2.884 kcal/mol; single-conformer vs
     Boltzmann-mean label differ by only |0.15| kcal → **Boltzmann relabel is a
     null** (homo confirmed it 2× independently, this is the 3rd).
  2. round10 AL "fix" ablation: training on 1,370 AL-hard pairs moved MAE
     **−0.03** (within noise).
  3. Three 3D-GNN architectures all landed at blend MAE 2.18 → real 3D geometry
     gives no clean gain over 2D + attentive pooling.
- **Status.** **The single-conformer r2SCAN-3c label-noise floor ≈ 2.9 kcal/mol.**
  The champion (2.215) is at it. Do not re-open targeted relabel debates without
  new evidence. The B97-3c lever (§1.5) is the sanctioned way to try to lower it,
  because it changes the *baseline*, not the label geometry.
- **Missing.** Whether the B97-3c lever actually clears the floor at full scale
  (= Tier B verdict).

### 1.8 Data provenance & recovery — 🔄
- **Principle.** gitignore'd large artifacts + no home backup = one storage
  policy change destroys them. The 2026-07 Snellius scratch purge did exactly
  that.
- **Status.** Systematic recovery since 2026-09-02: bit-exact rebuilds from
  `~/benzoin_backups` where possible (aldehyde Mordred, product Mordred restored
  2026-09-09, BDE); from-scratch recompute otherwise (cross r8/9 labels, ~41.5k
  SPs). rounds 1-7 reproduction validated (CV MAE 1.877 vs historical 1.883).
  Discipline now: `git add -f` results, `submit_backup_recovery_artifacts.sh`.
- **Missing.** Full homo DFT labels (~189k) confirmed unrecoverable → being
  recomputed. FILE_MAP.md long-standing uncommitted churn.

---

## 2. Prediction Optimization Workflow — the ML model

### 2.1 Learning framing — Δ-learning — ✅
- **Principle.** Instead of `model → ΔG` from scratch, predict the **correction**
  to a cheap physical estimate: `ΔG_pred = ΔG_baseline + f_ML(features)`, where
  `f_ML` learns `ΔG_DFT − ΔG_baseline`. The baseline (g-xTB, MAE 5.037) already
  captures most of the electronic-structure physics and all the trivial
  size/entropy scaling, so the ML target is a small, smooth, roughly
  mean-zero residual → an order of magnitude more data-efficient than direct
  regression, and it degrades gracefully (worst case ≈ the baseline).
- **Status.** Frozen. `BASELINE_COL` env-overridable (`CB_BASELINE_COL`) so the
  Tier B retrain can swap g-xTB → B97-3c with one flag.
- **Missing.** Nothing structural; the open question is *which* baseline (§1.5).

### 2.2 Featurization — 260 → 257 feature schema — ✅
- **Principle.** A pair is described by (a) **local QM descriptors** on the
  reactive atoms of each aldehyde — xTB frontier orbitals, Fukui indices,
  Mulliken/ADCH charges, Wiberg bond orders, QTAIM bond-critical-point
  properties, buried volume / Sterimol sterics — computed once per aldehyde and
  reused; (b) **Mordred** 2D/3D molecular descriptors (a curated slim subset,
  ~46% of the schema); (c) **RDKit 2D** descriptors (~19%); (d) product-side QM
  + BDE features; (e) hand-built `interaction_*` pair terms. The local-QM idea:
  the coupling chemistry happens at the formyl carbon, so charge / electrophilicity
  / steric bulk *there* should carry most of the signal.
- **Status.** Frozen 260-feature schema since round 7. 2026-09-08 audit → the 3
  `n_CHO` features are degenerate (>99% one value); dropping them (schema v2 =
  257) is A/B-harmless and applies at the post-Tier-B retrain.
  `interaction_*` terms: tried repeatedly, **none ever made the final list** (the
  GNN half learns those implicitly).
- **Missing.** Winsorizing the heavy-tailed `wbo_CC_new` / `mulliken_*` features
  (deferred to the Tier B retrain). No evidence a bigger feature set helps at
  ~21k rows (learning-curve flat).

### 2.3 Data splits — Bemis–Murcko scaffold-disjoint — ✅ (a hard-won correction)
- **Principle.** A random or molecule-level (InChIKey-disjoint) test set leaks:
  the test molecules' *scaffolds* are already in train, so the model interpolates
  and the reported error flatters real-world (novel-scaffold) generalization.
  The honest test is scaffold-disjoint: no Bemis–Murcko scaffold shared between
  train and test.
- **Result (2026-07-17).** The old molecule-level split had 93% of the frozen
  holdout's scaffolds already in train. A clean scaffold-disjoint 80/10/10 split
  (test n≈448, frozen since round 7) revealed a **real, reproducible +0.221 MAE
  (~9.8%) generalization gap**. Every historical "frozen holdout MAE" had
  over-stated true generalization by 0.2–0.5. The BDE sub-project hit the same
  trap independently (+43%). homo's own leakage premium is +15%.
- **Status.** Scaffold-disjoint is the only evaluation quoted. Any "frozen
  holdout MAE" must first be checked for scaffold overlap.
- **Missing.** Nothing.

### 2.4 Tabular models — XGBoost, MLP+XGB ensemble — ✅
- **Principle.** Gradient-boosted trees (XGB, depth-3, 300 trees, lr 0.05) are
  the strong baseline for tabular chemical descriptors — handle mixed scales,
  monotone-ish relationships, missing values, no scaling needed. The
  **MLP+XGB ensemble** (a scaler + MLP regressor + 2 XGB seeds, averaged) adds a
  smooth-function learner that catches signal trees miss; ~0.2 MAE better than
  single-XGB on the holdout.
- **Status.** `train_scaffold_disjoint.py` — single-XGB champion + ensemble,
  group-K-fold CV by `pair_key`. Holdout: single-XGB 2.548, ensemble 2.326.
- **Missing.** Retrain on the Tier B / schema-v2 table.

### 2.5 Graph model — triple-encoder attentive-pooling GNN — ✅
- **Principle.** Descriptors throw away bonding topology the model could exploit;
  a message-passing GNN reads the molecular graph directly. This one has
  **three encoders** (donor, acceptor, product graphs), attentive pooling
  (learned weighting of atoms into a molecule vector), and a head that also
  ingests the 257 QM descriptors (`x_d` channel) — so it is graph + descriptors,
  not graph alone.
- **Result.** On round 1-7 data the GNN blend was a **null**; from round 1-8
  (attentive pooling) it became robustly significant (P=0.992) and its blend
  weight has *risen* every round since (0.40 → 0.50). Lesson: architecture
  comparisons must be done at the target data scale — small-data nulls do not
  extrapolate.
- **Status.** `train_cross_gnn_arch_sweep.py` (arch=attentive, h128, l4, lr3e-4).
  Trained on CPU (the r1-10 champion GNN was CPU-trained). GNN-only holdout 2.313.
- **Missing.** Post-Tier-B retrain as a **4-seed ensemble** (2026-09-08: averaging
  4 seeds drops blend holdout MAE 2.215 → 2.137, w_gnn → 0.75; ~0.5 bootstrap-SE
  so not yet significant, but folds into the retrain).

### 2.6 Blending & seed-ensembling — ✅
- **Principle.** Two model families with decorrelated errors → a weighted
  average beats either. `w_gnn` chosen on the validation split (currently 0.50;
  ~0.75 with 4-seed GNN). Seed-ensembling reduces the GNN's ~±0.06 single-seed
  variance.
- **Status.** `predict_cross_champion.py` (the champion loader),
  `blend_gnn_seed_ensemble_r10.py`.
- **Missing.** Re-sweep `w_gnn` after the Tier B + 4-seed retrain.

### 2.7 Active learning — rounds 1–10 done, but 🅿️ PARKED / to be redone
- **Principle.** DFT is expensive; the useful question is which pairs to label
  next. Each round: score the unlabeled pool with a **pair-grouped bootstrap
  ensemble**, rank by prediction std (query-by-committee epistemic proxy),
  DFT-label the top batch, retrain.
- **What was done (2026-07-14 → 09-04).** 10 closed loops over the
  `candidates_v3` pool (~1.24 M pairs), producing the 35,528-pair labeled set
  and the r1-10 blend champion (MAE 2.215). The round10 ablation separated
  *diagnosis* from *fixing*: AL genuinely found blind spots (r1-9 MAE 2.78 on the
  picks vs 2.17 on holdout), but **training on 1,370 of them moved MAE −0.03**
  (within noise) → ~2k-scale rounds don't move the needle.
- **Status (user, 2026-09-10): the past AL is set aside for now, and cross AL
  will be redone.** Reason: the `candidates_v3` pool was **not the real chemical
  space** — the cross library was constructed wrong (§2.11). A fresh AL campaign
  will run over a correctly-defined pool (the flying dataset over the 220,860²
  space) once that and the improved labels (Tier B / B97-3c) are in place.
- **What is retained.** The 35,528 DFT-labeled pairs are still valid data. The
  r1-10 blend is kept as a **reference model**, not the forward line.
- **Missing.** A correctly-scoped candidate pool (§2.11); a fresh acquisition
  strategy decision (uncertainty vs multi-objective vs coverage) informed by the
  corrected space; the redone campaign.

### 2.8 Evaluation & uncertainty — ✅
- **Principle.** (a) Point error: scaffold-disjoint holdout MAE / R². (b)
  **Calibrated uncertainty**: split-conformal 90% prediction interval
  (distribution-free marginal coverage ≥ 90%; ±5.24 kcal, wide because the point
  estimate sits on the noise floor and residuals are heavy-tailed). (c)
  **Risk flags**: `baseline_risk` (a g-xTB-baseline-failure substructure —
  hypervalent P, sulfonyl, sulfoxide, nitro, N-oxide, Se, triflate; flagged rows
  have MAE 2.86 vs 2.10 → route to DFT), `dg_high_sigma` (OOD guard). (d)
  **Reformulation** (see 2.9): thresholding the same regressor's output as a
  favorable/unfavorable decision or a rank gives AUC 0.92–0.93, top-decile
  precision 0.64 — metrics *not* capped by the 2.9 kcal label floor, because a
  binary/ordinal call is robust to it by construction.
- **Status.** `build_predict_dg_calibration.py` → `predict_dg_calibration.json`
  (n=929); `eval_reformulation_classification_ranking.py`. All shipped 2026-09-07.
- **Missing.** Recalibrate after any champion retrain.

### 2.9 Deployment — `predict_dg.py` — ✅
- **Principle.** A single end-to-end entry point: featurize → assemble →
  prune to the schema → blend inference → emit point estimate + interval +
  decision columns, for any new aldehyde pair.
- **Status.** Working end-to-end (2026-09-07). Output columns: `dG_pred_kcal`,
  `ens_member_sigma`, `dg_favorable`/`dg_below_train_median`/`dg_rank_pct`,
  `dG_pi_lo_90`/`dG_pi_hi_90`, `baseline_risk`/`baseline_risk_motifs`,
  `dg_high_sigma`. SLURM path for the slow-geometry case wired.
- **Missing.** A fresh precision check on genuinely novel pairs (the historical
  "MAE 1.65 on 5 known pairs" predates a regression that was since fixed).
  Trap: any change to the shared `homo_v6/*_all.csv` schema must be followed by
  the `predict_dg.py` 20-pair smoke test (a 2026-09-06 shared-file bug silently
  broke it for all new pairs).

### 2.10 Sub-workflows

#### 2.10.a BDE prediction (bond dissociation energies) — ✅ champion trained
- **Principle.** Predict the BDE of the two bonds that matter for the coupling —
  the aldehyde formyl **C–H** and the product central **ketC–carbC** — as a
  target in their own right (a mechanistic handle), labels from the project's own
  g-xTB calculations.
- **Status.** champion **B6 = D-MPNN + local 3D descriptors via the `x_d`
  channel**; 2026-09-06 full-220k scaffold-disjoint retrain: **aldehyde MAE
  1.851 / product 2.826** (the old 42k-library 1.579/3.060 numbers are void —
  different data scale). Only robust architectural conclusion: `x_d` fusion
  beats pure-2D graphs (35–47 pts); finer changes are scaffold-leakage artifacts.
- **Missing.** Nothing active. Phase-3 3D BDE model: checked and **not built** —
  it actually predicts the cross-dG target and that architecture family is a
  confirmed null there. Entry point `pipeline/bde/STATUS.md`.

#### 2.10.b Homo-only ΔG model, from scratch — 🔄 (active, user-directed 2026-09-10)
- **Principle.** Rebuild the homo (A+A) model on the **full** library with
  recomputed DFT labels, using a **single XGBoost** and a **single attentive GNN**
  reported independently — deliberately *not* the iterated cross champion blend,
  to get a clean, understood full-data baseline.
- **Status.** DFT labels being recomputed (§1.5). Manifest split 179,431
  archived-geom + 4,621 regen. Three contamination bugs found & fixed 2026-09-10
  (orca_sp/ dir collision; `_extract` cross-chunk cache collision → 27% garbage
  dG; corrupt stored thermal → in-worker `--ohess` recompute). Campaign
  restarted clean. Old pre-purge homo champion (`ENSEMBLE72`, random-split MAE
  1.503) is still on-line for reference.
- **Missing.** Clean label campaign to drain (~2–3 days) → merge + QC verdict →
  `--full-library` assembler → the two models → results in
  `data/cross_benzoin/homo_standalone/README.md`.

#### 2.10.c Homo + cross unification (Rec-2) — 🟡 AMBER-GREEN, prep in progress
- **Principle.** If homo and cross ΔG are the same physics at matched conditions
  (work-set B showed they are — the apparent gap is split regime + data scale,
  not task difficulty), then pooling both into one model with an `is_homo` flag
  should help via more data / broader chemistry coverage.
- **Status.** `naive_merge` (homo:cross ≈ 1:1, 72-feat) gives a reproducible
  **−0.11 kcal** on both model classes; `finetune` is a null. Product Mordred
  restored (2026-09-09). Assembler `--full-library` mode pending.
- **Missing.** Wait for the from-scratch homo labels (2.10.b) → assemble a
  unified 260-feat table → retrain champion + GNN with `is_homo`/`sample_weight`
  → does −0.11 survive at 260-feat + GNN scale? If yes, add a GNN
  homo-pretrain→finetune path (the current champion GNN is pure-cross, never
  rebuilt post-purge).

#### 2.10.d Cheap-baseline lever — cross Tier B relabel (Rec-1 / work-set D) — 🔄
- **Principle.** §1.5: swap the Δ-learning baseline g-xTB → B97-3c. The pilot
  (residual std 4.32 → 1.11) says the Δ-model floor could drop ~4×.
- **Status.** Full self-consistent relabel of all 35,528 cross pairs running
  (~72% 2026-09-10). On drain: `merge_rec1_b973c_tierB.py` → QC verdict →
  `patch_train_table_tierB_b973c.py` (schema v2) → retrain
  `CB_BASELINE_COL=dG_b973c_kcal` → GNN 4-seed → blend re-sweep. Runbook:
  `data/cross_benzoin/rec1_b973c_tierB/DRAIN_RUNBOOK.md`.
- **Missing.** The drain + retrain + the verdict (does holdout MAE fall to
  ~1.0–1.5 = project-level breakthrough, or does the lever wash out at scale?).
  A partial-data preview showed holdout ens MAE 2.53 → 0.69 (control-isolated) —
  strong signal, not the headline yet.

### 2.11 Chemical-space definition & the "flying dataset" — ❌ needs building (user, 2026-09-10)
- **Principle.** A screening model is only as meaningful as the space it screens.
  Two corrections from the user:
  1. **homo space** = the 220,860 v6 aldehydes paired with themselves. Some are
     uncomputable by design (§1.2) — that's a recorded outcome, not a gap.
  2. **cross space** — the old `candidates_v3` pool (~1.24 M pairs) was a
     **wrongly-constructed subset**, not the real space. The real cross space is
     **every ordered pair of v6 aldehydes: 220,860² ≈ 4.88 × 10¹⁰** (~2.44 × 10¹⁰
     unordered; the donor/acceptor roles are chemically distinct so ordered is
     the honest count). `candidates_v3` should be retired as "the pool".
- **The "flying dataset" — the design.** You cannot and should not materialize
  48 billion product SMILES. Instead, a **lazy / virtual** dataset:
  - **Base index.** The v6 aldehyde library is the single source of truth,
    frozen with a canonical integer index `0 … 220,859` (stable, never an
    enumerate-on-the-fly index; = the library `index` column / InChIKey).
  - **A pair is an address**: `(donor_idx, acceptor_idx)`. No pair table on disk.
  - **Per-aldehyde caches** (compute-once, reuse-everywhere): geometry, xTB
    energies + thermal, local QM descriptors, Mordred, BDE — keyed by the base
    index. These already largely exist under `homo_v6/`.
  - **On-demand generation**: given a pair address, a generator produces the
    product SMILES (reaction template), assembles the 260-feature row from the
    two aldehyde caches + product-side features + pair/interaction terms, and
    (optionally) the g-xTB / B97-3c baseline — all lazily, streamed, cached at
    most transiently.
  - **Deliverable**: a documented spec + a thin read API
    (`pair(i, j) -> {smiles, features, baseline, split}`) so any future step
    (simulation, labeling, prediction, AL acquisition) reads the space uniformly
    without a giant file. Split assignment (scaffold-disjoint) is computed from
    the two aldehydes' scaffolds on the fly.
- **Status.** Not built. `CHEMICAL_SPACE.md` (spec) to be written.
- **Missing.** The spec; the canonical frozen aldehyde index; the read API; a
  decision on how DFT-label storage keys into it; retirement of `candidates_v3`.

---

## 3. Catalyst Space — OUT OF SCOPE for this project (2026-09-10, user)

**This project is the *substrate* axis only.** It predicts whether an aldehyde
pair gives a thermodynamically favorable benzoin (uncatalyzed ΔG). It does **not**
model the NHC catalyst, the kinetic barriers, or the enantioselectivity.

- **Boundary.** Read the model's favorable / rank output as "worth a closer
  look", not "will work" — a **thermodynamic pre-filter**. `predict_dg.py` docs
  should say so explicitly (small ❌ to add).
- **The catalyst side is a separate, mature effort** in sibling repos
  (`ElioChen/nhc-benzoin-pipeline`, `nhc-benzoin-active-learning`,
  `nhc-active-learning`, `stereo-catalyst-engine`, `nhc-pkah-predictor`):
  TS_CC / TS_CN per diastereomer → microkinetics → rate (Kozuch–Shaik energy
  span) + |ee|, with multi-objective pool AL over a ~13 M-stereoisomer NHC
  library. **Not managed here.** The `nhc-gsp-*` / `sourceB-kinetics` /
  `qm-benzoin` cluster jobs belong to it — do not touch them from this
  project's sessions.
- **Any substrate × catalyst integration is the user's / the NHC project's
  call, not this plan's.**

---

## 4. Cross-cutting infrastructure

| Component | Principle | Status | Missing |
|---|---|---|---|
| **Compute** | SLURM on Snellius: rome (521 nodes, big shared pool), genoa, fat_rome/fat_genoa (4.3× RAM, +50% billed, no QOS cap), gpu_h100/a100. QOS `MaxJobsPU=128` on rome+genoa. | ✅ working | Must **share nodes with the NHC project** on the same account (user directive 2026-09-10) — size `%N` with headroom, don't saturate a partition. |
| **Envs** | `venv/nhc-workflow` (pandas/xgb/sklearn/rdkit/mordred), `venv/nequip` (GNN CPU), `/home/schen3/xtb` (g-xTB-capable), `/home/schen3/orca` (`ORCA_SCF=default`). | ✅ | `envs/gnn` shared env corrupted (empty stdlib) — unused, ignore. |
| **Data storage** | Live repo `benzoin-dg-restored`; large artifacts gitignored → must be `git add -f` + archived. | 🔄 | Recovery from the 2026-07 purge ongoing (§1.8). |
| **Backups** | `~/benzoin_backups` survived the purge; `submit_backup_recovery_artifacts.sh` archives new results. | ✅ discipline in place | homo full DFT labels were never backed up → lost. |
| **Monitoring** | Session-scoped `Monitor` tasks (guardian: janitor + inode; campaign monitors per array). **Do not survive a session** — every handoff must rebuild them. | ✅ pattern | Inherent fragility; handoff §3 always lists rebuild commands. |
| **Handoff routine** | Every session-end: commit/push, write `HANDOFF_<date>.md` (fixed 8-section structure + autonomy clause), close `RUN_LOG`, update memory, emit a resume prompt. | ✅ | — |
| **Journal + plan** (this doc + `LAB_JOURNAL.md`) | User directive 2026-09-10: reasoned daily journal + this living plan; understand the principle of every step. | 🔄 just started | Backfill journal; keep both current. |

---

## 5. Settled decisions — do not re-litigate without new evidence

- cross **reference model** = r1-10 blend, MAE 2.215 (produced by the now-parked
  AL rounds 1–10). It is a baseline to beat, **not** the frozen forward line —
  cross AL is being redone on a correctly-defined space (§2.7, §2.11).
- **The `candidates_v3` pool was wrong.** The real cross space = 220,860² ordered
  aldehyde pairs; build the flying dataset (§2.11), retire `candidates_v3`.
- The **v6 aldehyde library includes uncomputable molecules by design** — a
  "can't compute" is a recorded result, not a bug.
- **Label-noise floor ≈ 2.9 kcal/mol** (single-conformer r2SCAN-3c). The
  reference model is on it.
- **No geometry-method bias** in the DFT labels (three-species ΔΔG ablation; the
  product-side g-xTB vs GFN2 bias cancels in the ΔG difference). Label-quality
  investigation is **closed**.
- **round10-scale AL is a null** for accuracy. Stop ~2k AL rounds.
- **Multi-conformer Boltzmann relabel is a null** (3× confirmed).
- **3D-GNN gives no clean gain** over 2D + attentive pooling (cross). Phase-3 3D
  BDE model not built.
- **homo is not harder than cross** at matched split + scale.
- **B97-3c baseline lever** is the sanctioned attempt to beat the floor (Tier B).
- **Scaffold-disjoint is the only honest evaluation.** Any frozen-holdout number
  is suspect until scaffold overlap is checked.
- Don't chase speed; understand each step (2026-09-10).
- Don't touch the NHC / kinetics / mace / orcasp / qm-benzoin jobs.

---

## 6. Immediate next actions (2026-09-10)

**Compute in flight (both throttled to share the cluster with the NHC project):**
1. **cross Tier B** 🔄 — B97-3c relabel, ~72%. On drain (monitor `bvuteg5xb`):
   `DRAIN_RUNBOOK.md` steps 1-6 → champion + GNN(4-seed) retrain with
   `CB_BASELINE_COL=dG_b973c_kcal`, schema v2. Verdict: does holdout MAE fall to
   ~1.0–1.5? *(This retrain is a better-labels experiment on the r1-10 data — it
   informs the reference model, not the redone AL.)*
2. **homo from-scratch labels** 🔄 — clean campaign after 3 contamination bugs +
   a 4th (per-atom thermal bound). Archived track (`submit_homo_sp.sh`, 179,431)
   + regen track (`submit_homo_regen.sh`, 4,621). On drain: `merge_homo_sp.py` →
   QC → `--full-library` assembler → single XGB + single GNN.

**Design / structure work (no compute, do carefully — user: understand every step):**
3. **Flying dataset spec** (§2.11) — write `CHEMICAL_SPACE.md`: freeze the
   canonical v6 aldehyde index, define the `pair(i, j)` read API, decide DFT-label
   keying, retire `candidates_v3`. Prerequisite for redone cross AL and for
   Goal-3 screening.
4. **Rec-2 unification** 🟡 — once homo labels land: unified table → retrain →
   does −0.11 survive at 260-feat + GNN?
5. **Catalyst Space scoping** — a conversation with the user: does this project
   extend into substrate × catalyst (§3.3), or stay the substrate pre-filter and
   hand off to `nhc-benzoin-pipeline`? What, if anything, gets integrated from
   the NHC repos (§3.2).
6. **Redone cross AL** — after (3) and better labels: decide acquisition strategy
   over the flying dataset, run a fresh campaign.

Keep `LAB_JOURNAL.md` current; close it each evening.
