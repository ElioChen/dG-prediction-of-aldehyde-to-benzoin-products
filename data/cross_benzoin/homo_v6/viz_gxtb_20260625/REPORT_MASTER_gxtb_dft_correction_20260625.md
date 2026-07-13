# MASTER REPORT — g-xTB → DFT ΔG correction for homo-benzoin (2026-06-25)

Consolidated record of the g-xTB→DFT correction study: data, method comparison, calibration
hierarchy, GNN architecture sweep, scope analysis, reactant-feature ablation, and the
dual-encoder test. All numbers are **interim** (DFT labels are from the in-progress full job).

---

## 0. Data provenance
- **Descriptors**: `data/cross_benzoin/homo_v6/products_all.csv` (220,724 homo products, 72 cols)
  + `aldehydes_all.csv` (reactant CHO-site descriptors) — job **24128375** `cb_feat`, drained 2026-06-24 ~06:02.
  - g-xTB AND GFN2-xTB ΔG present; 16 ADCH/QTAIM cols empty (MULTIWFN=0).
- **DFT labels (r2SCAN-3c, CPCM-DMSO)**: completed chunks of the **in-progress** full job **24178884**.
  - At report time ≈ **148–162k / 219k labeled (~67–74%)**, growing. Labels = front of manifest by index.
  - Representativeness checked: g-xTB ΔG & electronic-gap distributions of labeled vs unlabeled are
    near-identical (Δ≈0.05); only SASA +0.2σ (labeled slightly larger). **Interim results valid; refit on full labels when done.**

## 1. Method comparison (vs DFT, on labeled subset)
| method | MAE | bias |
|---|---|---|
| **g-xTB** | **4.32** | −2.54 |
| GFN2-xTB | 15.5 | −15.5 |
g-xTB is the correct Δ-baseline; GFN2 over-stabilizes the product by ~15 kcal. The earlier
"g-xTB vs GFN2 +13 kcal disagreement" is almost entirely **GFN2 error**, not g-xTB.

## 2. Correction hierarchy (Δ-learning DFT−g-xTB, 34 product QM features, test MAE)
| level | model | MAE | R² |
|---|---|---|---|
| L0 | raw g-xTB | 4.35 | 0.17 |
| L1 | +constant bias | 4.15 | 0.33 |
| L2 | affine | 4.03 | 0.37 |
| L3 | ridge(desc) | 2.87 | 0.63 |
| L4 | GBT(desc) | **2.46** | 0.70 |
Most raw error is a fixed offset; descriptor model needed for the structured remainder.
Top GBT drivers: `P_int` (dispersion-surface, ~31%), `wbo_CC_new`, `wbo_CO_ket`, `pa_ketO`, `xtb_EA`.

## 3. GNN architecture study (6 archs, all-categories, Δ-learning, test MAE)
| arch | MAE | R² | note |
|---|---|---|---|
| **gine_hybrid** (graph+34 QM) | **2.13** | 0.76 | only one to beat GBT |
| gine_big | 2.65 | 0.66 | capacity doesn't help |
| gine | 2.66 | 0.66 | pure-graph ceiling |
| nnconv (D-MPNN-like) | 2.67 | 0.66 | |
| gat | 2.72 | 0.65 | |
| gcn | 2.80 | 0.63 | drops bond features |
**Pure-graph plateaus ~2.65–2.80 regardless of conv operator; QM descriptors at readout break it.
Bottleneck is information, not architecture** (2D graph can't see 3D/electronic state that drives g-xTB error).

**Full-data GNN re-run (job 24224265/245, n=219k, same split) confirms the partial-n results:**
pure-graph gine 2.60 / gine_big 2.58 / gat 2.58 / gcn 2.68 / nnconv 2.62; gine_hybrid(+34QM) **1.991**;
dual_qm(+56QM) **1.646**. Ranking unchanged: tabular ensemble 1.566 < dual_qm 1.646 < gine_hybrid 1.99 <
pure-graph 2.58-2.68. Logged to MLflow exp1_gnn_arch_full / exp1_gnn_dual_full.


## 4. Scope analysis (Ridge/MLP/GBT × {all, aromatic, aliphatic}, test MAE)
| scope | n | raw | Ridge | MLP | GBT |
|---|---|---|---|---|---|
| all | 152,781 | 4.21 | 2.77 | 2.34 | 2.38 |
| aromatic | 97,520 | 4.20 | 2.37 | **1.91** | 1.95 |
| aliphatic | 54,253 | 4.12 | 3.17 | 2.88 | 2.81 |
- Aromatic much easier (homogeneous chemistry); aliphatic is the hard third dragging "all" up.
- **Specialization is NOT the win**: on the same aromatic test, aromatic-specialist 1.95 vs
  all-trained generalist 2.01 (only 3% gap). Keep all-category training (scope decision [aromatic-only dropped]).
- **gine_hybrid on aromatic-only: TEST MAE 1.673, R² 0.788** (confirms ~1.6–1.7 ceiling).

## 5. ⭐ Reactant-feature ablation (the key finding)
ΔG_error = G_product_error − 2·G_aldehyde_error, so product-only features miss the reactant side.
Adding `aldehydes_all` CHO-site descriptors (join via donor_id), n=162,177:
| feature set | nfeat | GBT | MLP |
|---|---|---|---|
| product-only | 34 | 2.35 | 2.30 |
| **product + reactant** | 56 | **1.87** | **1.73** |
| reactant-only | 22 | 2.71 | 2.69 |
**Adding reactant descriptors: −0.5 to −0.6 kcal MAE (−20–25%).** product+reactant MLP (1.73)
beats the GNN hybrid (2.13) and GBT (2.46). **Information completeness (both sides) > model architecture.**

## 6. Dual-encoder Δ-GNN (graph-level "both") — job 24212556 (n=166k)
Product graph + aldehyde graph, combined as `[h_P, h_A, h_P − 2·h_A]` (thermodynamic cycle).
| config | test MAE | R² |
|---|---|---|
| product_only (1 graph) | 2.555 | 0.658 |
| dual (product + aldehyde graphs) | 2.540 | 0.661 |
| **dual_qm (dual graphs + 34 QM)** | **2.017** | 0.773 |
**Key:** adding the aldehyde GRAPH alone barely helps (2.555→2.540) — for homo, the product
graph already contains the aldehyde substructure, so reactant *topology* is redundant. The
reactant value is in its *QM descriptors* (isolated electronic state), not its graph: dual_qm
(2.02) edges out product+QM gine_hybrid (2.13) but still loses to the 56-feature MLP (1.73).
**Confirms section 5: information (reactant QM) > architecture.**

**Follow-up — dual_qm with the FULL 56 QM features at readout (n=202k):** product_only 2.46,
dual(no QM) 2.47, **dual_qm(56) 1.616 / R²0.84** — i.e. once given information parity, the GNN
*exactly matches* the MLP+XGB tabular ensemble (1.61). The graph encoder itself contributes ~0
(2.46 without QM); all the lift is the 56 QM at readout. Definitive: **a GNN can match but not
beat tabular here — the bottleneck is information, not architecture.** (So production uses the
cheaper CPU tabular ensemble.)

## 6b. Tabular headroom & ensemble (56 features)
More-complex tabular gives diminishing returns; the **MLP+XGB 0.5/0.5 ensemble** is the lever:
| model (all, n=175k) | MAE | R² |
|---|---|---|
| MLP(256,128) | 1.71 | 0.834 |
| MLP(512,256,128) deeper | 1.72 | 0.833 |
| XGB(1200,d6) | 1.77 | 0.827 |
| XGB(1500,d8) | 1.69 | 0.837 |
| **ENSEMBLE MLP+XGB** | **1.67** | 0.841 |
Deeper nets don't help — tabular is near-saturated; beating ~1.67 needs **new information** (→ 3D GNN).

### Scope × ensemble (56 features, n=186k) — ensemble wins in every scope
| scope | n | MLP | XGB | **ENSEMBLE** |
|---|---|---|---|---|
| all | 186k | 1.69 | 1.68 | **1.611 / R²0.851** |
| **aromatic** | 124,766 | 1.48 | 1.45 | **1.405 / R²0.853** |
| aliphatic | 60,429 | 2.15 | 2.06 | **2.031 / R²0.809** |
Ensemble beats both base models in all scopes; aromatic ensemble **1.405** is the best subset result
(vs aromatic gine_hybrid 1.67, aromatic 34-feat MLP 1.91). Aliphatic remains the hard third.

## 6d. Architecture comparison & the SMILES→ΔG deployment goal
Tabular ensemble (56QM) and GNN dual_qm (56feat) both reach **~1.61** — functionally equivalent.

| dimension | Tabular ensemble (MLP+3×XGB) — PRODUCTION | GNN dual_qm |
|---|---|---|
| input | 56 QM scalars (flat) | 56 QM at readout + product graph + aldehyde graph (GINE) |
| where power comes from | the 56 QM | **~all from the 56 QM** (graph-only=2.46; +QM→1.62; graph marginal≈0) |
| regressor | XGBoost (axis splits/interactions) + MLP, averaged | graph-embed ⊕ 56QM → MLP head (graph embed ~redundant) |
| cost | **CPU, min train, ms infer, KB-MB** | GPU, ~30min, 516k params, PyG at infer |
| interpretability / uncertainty | high (SHAP) / ensemble-variance free | low / needs ensembling |

**Key reframing for the end goal (input SMILES → output ΔG):** the accurate model is a Δ-*correction*
on QM descriptors, **not** a pure SMILES→ΔG map. A real SMILES→ΔG product needs the full front-end:
```
SMILES → RDKit embed → funnel_v3 conformer → xTB (g-xTB ΔG + 56 QM desc)
       → +16 RDKit global (instant) → ENSEMBLE Δ → ΔG = g-xTB + Δ  (+ uncertainty, route_to_dft)
```
Per-molecule cost is dominated by **conformer search + xTB (~minutes)**, NOT the model (ms). So the
model-architecture choice barely affects deployment cost — the QM front-end does. This implies a **two-tier
hierarchy** (cf [modeling-direction]):
- **Tier-1 triage (pure SMILES→ΔG, no QM):** a 2D-graph/2D-descriptor model predicting ΔG directly,
  ms/mol, MAE ~2.5–2.8 — screen millions cheaply. **Here a GNN is the right tool** (tabular has no input without QM).
- **Tier-2 accurate (this work):** SMILES→geometry→xTB→QM→ensemble-Δ, ~min/mol, **MAE 1.61 / confident-85% 1.43**.
  Here tabular = GNN, so use the **cheap CPU tabular ensemble**; the GNN's graph machinery is dead weight.

**Product contract**: return `(ΔG_corrected, uncertainty_std, route_to_dft)` — confidence-aware, with the
unreliable ~15% (P/B/S / large-flexible) routed to DFT. Ship via `src/benzoin_dG/predict.py` wrapping the chain.

## 6c-v2. PRODUCTION model UPGRADED (2026-06-26) — +deep XGB + quantile-PI routing
[finalize_correction.py](../../../pipeline/analysis/finalize_correction.py) → `gxtb_dft_correction_ENSEMBLE72_20260626.joblib`
- **Point**: ensemble MLP(512-256-128) + XGB(d8) + **XGB(d10)** on 72 feat → **test MAE 1.566, R² 0.861** (was 1.646).
- **Routing**: **quantile prediction-interval width** (exp3) instead of ensemble-std → confident-85% MAE
  **1.280** (was 1.449), routed 3.19, **separation 2.49×** (was 1.72×). Output col `uncertainty_pi_width`.
- **Train/val/test viz** (`model_figures/`, 70/20/10 = 152673/43621/21811): XGB_d10 overfits (train 0.56 ≪
  test 1.59); the ensemble keeps the same test but a healthier train 1.04 → **ensemble preferred for robustness**.
  Per-model parity PNGs + `all_models_train_vs_test.png` + MLflow exp `gxtb_dft_viz_fullmetrics` (train/val/test logged).


**Optuna XGB tuning (exp6, optimize val MAE):** best (depth9, min_child_weight24, reg_lambda31.9) → train 0.436 / val 1.553 / **test 1.557** (best single model). Reframes the 'overfit' concern: the test-optimal model has the LOWEST train MAE — the low-train/high-test gap is model capacity, NOT a generalization problem (val≈test, scaffold-split fine). Don't force a smaller gap (hurts test). Candidate to swap into the production ensemble's XGB.

## 6c. FINAL production model (2026-06-26, **100% labels** via auto-chain) — ensemble72 + uncertainty routing
[finalize_correction.py](../../../pipeline/analysis/finalize_correction.py) → `pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`
- **Model**: MLP + 3×XGB ensemble on **72 features** (56 QM + 16 product RDKit global). Δ-learning.
- **TEST MAE 1.646, R² 0.851** on **219,364/219,421 (100%) labels** (n=219,095 full-feature).
  - The 97%→100% refit nudged MAE 1.612→**1.646** because the ~6k recovered timeout molecules are
    exactly the hard P/B/S/large-flexible tail; **1.646 is the honest full-distribution number.**
- **Uncertainty routing** (ensemble std): flag most-uncertain 15% → route to DFT.
  - confident 85% **MAE 1.449** | routed 15% MAE 2.76.
- DFT labels consolidated to single source `data/raw/dft_sp_funnelv3/dft_labels_all.parquet`
  (219,421 rows; 219,364 valid; only 57 hard failures left after 7200s retry).
- Full-library output `products_dG_corrected_FINAL_20260626.csv`: per-mol `dG_gxtb_corrected_final` +
  `uncertainty_std` + `route_to_dft`.
- **Ablations that informed this**: reactant 2D-global descriptors add nothing (product topology
  already implies them; n=211k: 56+prodG 1.632 vs +reactG 1.638) — only product global kept.
  Routing separation is moderate (2.63 vs 1.43); future: better-calibrated uncertainty (evidential/AD).
- Auto-finalize chain (job 24218826, dep on DFT arrays 24178884+24191120) re-runs this at 100% labels.

## 6e. UNIFIED full-dataset benchmark (n=219,095, same 70/20/10 seed-42 split) → MLflow
All models finally on identical complete data (earlier numbers were partial-n / different scripts).
Logged to `sqlite:///mlflow_benchmark.db` (exp `gxtb_dft_full_benchmark_20260626`). View:
`mlflow ui --backend-store-uri sqlite:////scratch-shared/schen3/benzoin-dg/mlflow_benchmark.db`.

| model | test MAE | R² |
|---|---|---|
| **Ensemble(MLP+XGB) 72feat** | **1.614** | 0.855 |
| GNN dual_qm 56feat (ref) | 1.616 | — |
| Ensemble 56feat | 1.648 | 0.848 |
| XGB 72feat | 1.679 | 0.845 |
| MLP 72feat | 1.686 | 0.843 |
| XGB 56feat / MLP 56feat | 1.717 / 1.727 | 0.84 |
| 3D DimeNet++ (60k ref) | 2.04 | — |
| Ridge 72feat | 2.105 | 0.782 |
| GNN gine_hybrid (ref) | 2.13 | — |
| Ensemble/XGB 34feat | 2.24 | 0.714 |
| Tier-1 pure-SMILES (ref) | 2.75 | — |
| Ridge 34feat | 2.80 | 0.609 |
| raw g-xTB / raw GFN2 | 4.26 / 15.70 | — |
On full data the 16 global descriptors give a real (small) edge (56→72 ensemble 1.648→**1.614**);
GNN dual_qm ties the tabular ensemble. Feature-set value (34→56→72) dwarfs model choice within a set.
Extended variants (Linear/Ridge/Lasso, MLP/XGB depths), GNN 6-arch full, scaffold-split, quantile
uncertainty, Tier-1 distillation logged to MLflow exps 1–5 (jobs 24224242-266).


## 6f. Label-noise floor quantified (conformer-noise experiment, job 24225782, 34 mols × 5 confs)
r2SCAN-3c on multiple conformers per product (only product conformer varies → measures ΔG label noise):
- conformer ΔG **std mean 2.68 / median 2.28 kcal**, range 7.4 (inflated by non-thermal high-E confs);
- **lowest-E-conformer (funnel_v3's choice) vs Boltzmann-ensemble = 0.35 kcal systematic bias** (cleanest number).
→ The single-conformer label approximation carries ~0.35–1+ kcal uncertainty — a substantial slice of the 1.56
MAE floor. **The model has hit the LABEL noise floor**; lowering it needs conformer-ensemble (Boltzmann) DFT
labels, not better models (proven exhausted). Caveat: rough estimate (RDKit confs ≠ funnel_v3 CREST; std includes
non-populated confs). Data: `conformer_noise_summary.csv`.

## 7. Current best & takeaways
- **CHAMPION: MLP+XGB ensemble on product+reactant 56 features — all-category test MAE 1.611, R² 0.851**
  (aromatic 1.405, aliphatic 2.031). CPU-only, no GPU. Wired into the auto-finalize chain.
- Two biggest levers were both **information**, not architecture: +QM descriptors (2.66→2.13), +reactant descriptors (2.35→1.73); ensemble shaves a further ~0.07.
- GNN ladder: pure-graph 2.55-2.80 < gine_hybrid 2.13 < dual_qm 2.02 — all lose to the 56-feat tabular.
- **3D GNN tested (60k subset, z+pos only) — all LOSE to the same-data 56-feat MLP (1.83):**
  SchNet 2.27, **DimeNet++ 2.04** (best 3D; angles help over SchNet), ViSNet diverged (numerical, not tuned).
  Ran in the `nequip` venv (torch_cluster native + torch_sparse pip-installed for DimeNet++); the `gnn`
  env needed a pure-torch `radius_graph` patch (no torch-cluster) and can't run DimeNet++ (no torch-sparse).
  Hand-crafted descriptors already distill the dispersion/electronic physics; geometry-from-scratch
  needs more data/electronic node features. **Charge-augmented SchNet ablation in flight (job 24217925).**

## 8. Caveats
- Interim numbers were on partial DFT labels; **final model refit at ~100% via auto-chain** (DFT arrays
  24178884+24191120 drained 2026-06-26; retry7200 24219990 recovers ~6.2k timeouts; finalize 24220073 refits).
- 16 product + 7 reactant ADCH/QTAIM families were empty full-library (MULTIWFN=0); a 2.5k validation subset (job 24219571, P/B/S-enriched) was computed on saved geometries.
  **RESULT: ADCH/QTAIM does NOT help** — 5-fold CV on 2483 mols: 56-QM 2.437 vs 56+ADCH/QTAIM(23) 2.451 (Δ+0.014;
  P/B/S subset +0.024). Even reactant ADCH/QTAIM (99-100% filled) + product QTAIM (100%) add nothing. **Decision:
  do NOT back-fill ADCH/QTAIM full-library (~2.5-5k core-h saved).** Closes the 'more descriptors' question:
  reactant-2D-global, 3D-GNN, and ADCH/QTAIM all fail to break the ~1.6 floor — 56 QM (+marginal global) is the information ceiling.

## 8b. ROADMAP / VISION — production "SMILES → ΔG" (forward-looking plan, not yet built)
Goal: a deployable `predict(SMILES) -> (ΔG, uncertainty, route_to_dft)`. The accurate Tier-2 model is a
Δ-correction on QM descriptors, so it needs a per-molecule xTB+geometry front-end (~min/mol). To serve
millions, plan a **two-tier cascade + distillation** (complementary, NOT a replacement of the current models):

```
millions of SMILES
   │  TIER-1  pure SMILES→ΔG (Morgan2048 + 16 RDKit-2D global → XGB); ms/mol; **MAE 2.75 / R²0.59** (BUILT, n=214k; aromatic 2.49 / aliphatic 3.32)
   ▼  cheap pre-filter / rank, drop the clearly-bad
shortlist
   │  TIER-2  SMILES→funnel_v3→xTB(g-xTB ΔG+56 QM)→ensemble-Δ; min/mol; MAE 1.61 / confident-85% 1.43  [BUILT]
   ▼  accurate rank + per-mol uncertainty
most-uncertain ~15% (route_to_dft)
   │  DFT r2SCAN-3c  (ground truth)
   ▼
final ΔG
```
**Why the layers don't conflict** (the current MLP+XGB / GNN ARE Tier-2, untouched):
- different cost/accuracy regimes (ms vs min); used together as a funnel.
- shared assets: same DFT labels, same 16 RDKit-2D global descriptors; no duplicate effort.
- mutually reinforcing: Tier-1 confidence decides who is worth escalating to Tier-2; Tier-2 (or DFT)
  cheaply labels masses → **distill** a stronger Tier-1 (teacher→student) → iterate.
- architecture choice falls out cleanly: **Tier-1 = GNN/2D (only option without QM); Tier-2 = CPU tabular
  ensemble (≡ GNN dual_qm at 1.61, but cheaper).**
Status: Tier-2 built (ensemble72 + routing). Tier-1 BUILT (tier1_smiles_to_dg.py): pure-SMILES MAE 2.75 (vs Tier-2 1.61) — 70% worse but ~1000× faster, ideal triage. Open:
distillation loop, scaffold-split generalization test, optional ADCH/QTAIM back-fill if the subset validates.

## 9. File index (all in `viz_gxtb_20260625/` unless noted)

## 9. File index (all in `viz_gxtb_20260625/` unless noted)
- Reports: `REPORT_gxtb_products_full_*.md`, `REPORT_gxtb_to_dft_calibration_*.md`, **this master report**.
- Metrics: `gxtb_to_dft_metrics.csv`, `gnn_arch_results.csv`, `gnn_arch_results_aromatic.csv`,
  `scope_arch_results.csv`, `ablation_reactant.csv`, `gnn_dual_results.csv` (pending).
- Figures: `01–13` (deep viz), `20–24` (calibration), `26–28` (GNN/scope).
- Full-library corrected ΔG: `products_gxtb_gnn_corrected.csv` (gine_hybrid).
- Scripts (in `pipeline/analysis/`): `viz_gxtb_products_full.py`, `calibrate_gxtb_to_dft.py`,
  `gnn_arch_study_gxtb_dft.py`, `scope_arch_compare.py`, `ablation_reactant_features.py`, `gnn_dual_encoder.py`.
- Descriptor meanings: `pipeline/docs/DESCRIPTOR_DICTIONARY_products.md`.
