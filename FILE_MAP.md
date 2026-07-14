# FILE_MAP — where things live & what they're for

> **Living index.** Keep this in sync: whenever a script, dataset, model, figure,
> or output dir is created/moved/retired, update the relevant row here in the same
> change. Goal: never lose track of a file or directory again.
>
> **Conventions enforced here**
> - **Plotting scripts are kept, never throwaway.** Every figure must be produced by a
>   committed `.py` that is registered in the *Figures & plotting scripts* table below
>   (script path → output path → purpose).
> - **One standalone figure per file** (no multi-panel composites).
> - **Never delete/overwrite prior outputs** — new results go to new filenames.

_Last updated: 2026-07-14._

## Roots
| Path | What | Notes |
|---|---|---|
| `/scratch-shared/schen3/benzoin-dg` | **Active package** (git repo) | Same fs as `/gpfs/scratch1/shared/schen3/benzoin-dg` (symlink). All current work. |
| `/gpfs/scratch1/shared/schen3/ml` | **OLD pre-package area** | Legacy (`train_delta.py`, `sweep_delta.py` etc. from Jun 11–12). `labels_expand300` + `desc_expand` deleted 2026-06-23 (superseded, inode cleanup). |
| `/gpfs/scratch1/shared/schen3/envs/gnn/bin/python` | **Python env** | rdkit + xgboost + optuna + mlflow + shap + chemprop. Used by all training/GNN/analysis. |

## Code — `pipeline/`
| Path | Purpose |
|---|---|
| `pipeline/sweep_delta.py`, `delta_core.py`, `train_delta.py` | Δ-learning (tree) model: Optuna sweep + core + train |
| `pipeline/assemble_model.py` | Assemble winning Δ-model + AD reference → `src/benzoin_dG/models/` |
| `pipeline/train_surrogate.py` | 2D-SMILES surrogate model |
| `pipeline/retrain_report.py` | Retrain + report |
| `pipeline/analyze_*.py` (outliers, energies, screen_v5, rejected_v4, benzoin_deep) | Analysis scripts |
| `pipeline/analysis/library_homo_v6_analysis.py` | homo_v6 220k library: completeness + ΔG screen + g-xTB vs xTB (figs+report) |
| `pipeline/analysis/viz_gxtb_products_full.py` | DEEP g-xTB vs GFN2 dive on products_all.csv: parity/residual + disagreement-driver & descriptor correlations → `data/cross_benzoin/homo_v6/viz_gxtb_20260625/` (13 PNG + REPORT) |
| `pipeline/analysis/viz_dG_dft_vs_semiemp.py` | RAW ΔG correlation parity: DFT r2SCAN-3c (y) vs GFN2 & g-xTB (x), n=213k (dft_sp_funnelv3 ⋈ products_all) → `viz_dG_corrected_20260629/` 08_parity_DFT_vs_gfn2.png, 09_parity_DFT_vs_gxtb.png + REPORT. g-xTB MAE 4.15 / R²0.36 vs GFN2 15.63 / R²0.34 |
| `pipeline/analysis/viz_dG_corrected_final.py` | dG plots for the Jun-26 FINAL corrected deliverable `products_dG_corrected_FINAL_20260626.csv` (218k): corrected-ΔG dist, raw-vs-corrected shift, correction mag, PI-width uncertainty, ΔG-vs-uncertainty hexbin, ΔG-by-routing, cumulative exergonicity → `data/cross_benzoin/homo_v6/viz_dG_corrected_20260629/` (7 PNG + REPORT). median ΔG 3.52→5.44; 15% route_to_dft |
| `pipeline/analysis/calibrate_gxtb_to_dft.py` | g-xTB→DFT(r2SCAN-3c) correction hierarchy (L0 raw→L4 GBT Δ), 70/20/10; test MAE 4.35→2.46 vs GFN2 15.6 → same viz_gxtb_20260625/ (5 PNG + REPORT + metrics.csv) |
| `pipeline/docs/WORKFLOW_homo_v6_simulation.md` | Plain-English homo_v6 (220k) simulation workflow, geometry-optimization focused: 6-stage conformer funnel (reproducible ETKDG embed → GFN-FF relax → GFN2 single-point keep-10 → GFN2 opt → connectivity guard → lowest) + GFN2 free energy + g-xTB single-point + reaction ΔG + full descriptor guide + code-location table |
| `pipeline/docs/DESCRIPTOR_DICTIONARY_products.md` | Chemical meaning of every products_all.csv descriptor (ketC/carbC/hydO atoms, electronic/charge/WBO/Fukui/steric/H-bond families) |
| `pipeline/docs/REPORT_MASTER_gxtb_dft_correction_20260625.md` | **MASTER** g-xTB→DFT study: method cmp + calibration L0-L4 + 6-arch GNN + scope + reactant ablation (56-feat MLP 1.73) + dual-encoder; mirror in data viz dir + /home backup |
| `pipeline/analysis/gnn_delta_gxtb_dft.py` + `slurm/submit_gnn_delta_gxtb.sh` | GINE Δ-GNN predicting (DFT−g-xTB) from product SMILES, n≈136k; vs GBT 2.46; predicts full-library corrected ΔG → viz_gxtb_20260625/ (job 24205672) |
| `pipeline/analysis/gnn_arch_study_gxtb_dft.py` + `slurm/submit_gnn_arch_study.sh` | 6-arch GNN sweep (gine/gine_big/gat/gcn/nnconv/**gine_hybrid**); gine_hybrid WINS test MAE **2.13** R²0.76 (beats GBT 2.46); pure-graph ~2.65-2.80 lose; → 26/27 PNG + gnn_arch_results.csv |
| `pipeline/analysis/scope_arch_compare.py` + `aldehyde_class.parquet` | Ridge/MLP/GBT × {all,aromatic,aliphatic} Δ-correction; aromatic MLP **1.91** vs all 2.34 vs aliphatic 2.88; specialist≈generalist on aromatic (1.95 vs 2.01) → 28_scope_arch_compare.png |
| `pipeline/docs/REPORT_homo_v6_gnn_architectures_20260629.md` | GNN ARCHITECTURE GUIDE: shared backbone + each conv operator explained (gine/gine_big/**gat** GATv2/**gcn** no-edge ablation/**nnconv** edge-conditioned D-MPNN) + gine_hybrid/dual_qm/3D (SchNet/DimeNet++); all plateau ~2.6 (2D)/~2.0 (3D), only QM-injection reaches tabular 1.61 → info > architecture |
| `pipeline/analysis/plot_model_feature_importance.py` | XGBoost GAIN feature-importance for the key 72-feat Δ-models, loaded from SAVED bundles (no retrain); family-colored bars → `viz_gxtb_20260625/` 30_featimp_xgb_d8.png, 31_featimp_xgb_d10.png, 32_featimp_xgb_optuna.png. Top driver P_int ~18-20%, then reactant ald_pa_CHO_O / ald_wbo_CO / ald_P_int |
| `pipeline/build_cb_training_table.py` | Build Δ-training table: DFT labels ⨝ cross-benzoin aldehyde+product QM descriptors (`--baseline gfn2\|gxtb`) |
| `pipeline/predict_library.py` | Batch-score the 220k homo library: `dG_pred = dG_gxtb + cb-Δ-model(ald_/prod_ feats)`; joins by id |
| `pipeline/slurm/submit_cb_train.sh` | SLURM: cb Δ-sweep on `featurize_cb_homo_train_gxtb.parquet`, sweep-only (does NOT ship over production) → `runs_cb_gxtb/models/` |
| `pipeline/train_cb_721.py` | Δ-train with 70/20/10 train/val/test holdout (not K-fold), ALL categories, g-xTB baseline |
| `pipeline/learning_curve_cb.py` | Learning curve (test MAE vs #labels) to principally size the DFT label set → `data/analysis/learning_curve_cb/` |
| `pipeline/select_subset_maxmin.py` | Principled ~1500 training subset on v6: category-stratified MaxMin diversity → `data/analysis/subset_v6/subset_v6_maxmin_1500.csv` |
| `pipeline/compute/dft_sp_from_geom.py` | **DRAFT** efficient r2SCAN-3c SP on SAVED funnel_v3 geoms (skips conf-search/xTB-opt, reuses xTB thermal); node-local scratch |
| `pipeline/slurm/submit_dft_sp_funnelv3.sh` | **DRAFT** genoa-tuned submit for the above (48c/48 workers, %128, CSV-only) — NOT launched |
| `data/analysis/subset_v6/` | v6-library subset re-selection (fingerprint k-means; auto-k=400, weak silhouette) |
| `pipeline/plot_*.py` (parity_panels, baseline_ab, conformer_xtb_dft_conclusions) | Plotting scripts (register in Figures table) |
| `pipeline/gnn/gnn_core.py`, `train_gnn.py`, `sweep_gnn.py` | Hybrid D-MPNN (Chemprop) |
| `pipeline/analysis/library_v6_viz.py` | Structural library (filter_v6) composition figures |
| `pipeline/analysis/analyze_dft_sp.py`, `screen_v6_funcgroup_analysis.py` | DFT-SP & screen functional-group analysis |
| `pipeline/docs/REPORT_screen_v6_models_20260629.md` | CONSOLIDATED: all screen_v6 (GFN2, 1st full-lib) surrogate models of `dG_xtb_kcal` — linear 0.55 → MLP/GBT ~0.66 (aromatic) ceiling; ADCH/QTAIM +0.01; scaffold-split (rare 0.54); GINE GNN 0.611 < GBT 0.631 (2D-graph loses) + GINE explainer. NOT Δ-learning |
| `pipeline/slurm/submit_train.sh` | SLURM: Δ-train (genoa CPU) |
| `pipeline/slurm/submit_gnn.sh`, `submit_gnn_dG.sh` | SLURM: GNN (gpu_a100); `MODE=sweep,TRIALS=40` for full sweep |
| `pipeline/legacy/` | Deprecated scripts (e.g. `add_expand_chunk.py`) — do not use |
| `pipeline/docs/*.md` | Findings/analysis notes |
| `pipeline/RUNBOOK_v6_products.md` | Runbook: full-library cross-benzoin featurize + post-array concat |
| `src/benzoin_dG/predict.py`, `models/` | Shipped package: inference + shipped model |
| `cross_benzoin/cb_featurize.py` | Unified funnel_v3 aldehyde+product featurizer (`--emit-aldehydes`) |
| `cross_benzoin/ARCHITECTURE.md` | Cross-benzoin package design |
| `cross_benzoin/slurm/submit_cb_featurize_array.sh` | SLURM array driver for featurize |
| `cross_benzoin/slurm/fulllib_watch.sh` | Long-poll watch: exits on array drain / quota / mass-fail (re-arm as tracked bg task) |
| `cross_benzoin/prepare_pair_chunks.py` | Stream a directed candidate CSV.gz into bounded `cb_featurize.py --pairs` manifests |
| `cross_benzoin/prepare_product_manifest.py` | Cheap directed product-SMILES enumeration and chemistry QC before conformer/QM work |
| `cross_benzoin/docs/CROSS_BENZOIN_ML_RECOMMENDATIONS.md` | Homo-to-cross transfer, fusion validation, delta-learning and scaling recommendations |
| `cross_benzoin/docs/DESCRIPTOR_POLICY_CROSS.md` | Donor/acceptor/product/interaction descriptor responsibilities, pruning and ablation rules |
| `cross_benzoin/docs/NEXT_STEPS.md` | Phased cross-benzoin roadmap with acceptance gates and package/API promotion criteria |
| `cross_benzoin/select_cross_pilot_sample.py` | Draws a stratified, aldehyde-cache-hit-only directed cross pilot sample (6 category combos, both orientations, no self-pairs) straight from `aldehydes_clean_v6.csv`, no LFS needed |
| `cross_benzoin/docs/REPORT_codex_gap_analysis_20260714_{EN,ZH}.md` | Gap analysis: codex's GitHub-side cross-benzoin session vs. actual repo state; documents the PR-merge data-loss bug found+fixed, README staleness fix, g-xTB silent-failure fix, the array resume-skip bug found+fixed, and the first real cross-benzoin pilot compute run |
| `cross_benzoin/docs/METHODS_problem_definition_and_math_{EN,ZH}.md` | **Paper-ready** problem definition + math: ΔG_rxn vs ΔG‡, multi-fidelity Δ-learning decomposition, per-species-correction ML target, Boltzmann ensemble free energy, role-aware descriptor construction, directed-vs-unordered-pair rationale (empirically confirmed by cross_pilot_v1), 4-regime evaluation protocol, split hierarchy |

## Data — `data/`
| Path | Purpose |
|---|---|
| `data/featurize.parquet` | **ACTIVE training table** (Jun 15) — used by `submit_train.sh` / `submit_gnn.sh` |
| `data/featurize_funnelv3*.parquet`, `_gxtb_baseline`, `_v2`, `_funnel`, `_legacy_sparse`, `*.bak` | Dataset variants/backups (not the active one) |
| `data/featurize_cb_homo_train_{gfn2,gxtb}.parquet` | **NEW** Δ-training tables (2900 rows): DFT labels + cross-benzoin aldehyde+product QM descriptors, no RDKit. Baseline = GFN2 vs g-xTB. Built 2026-06-24 |
| `data/analysis/library_homo_v6/<ts>_*.png + _REPORT.md` | homo_v6 library analysis figures + report |
| `data/labels/`, `data/descriptors/` | Active label + descriptor inputs (package, self-contained) |
| `data/labels_pbe0_old/`, `descriptors_pbe0_old/`, `labels_partial/` | Old/partial label sets |
| `data/library/` | Structural library: `aldehydes_clean_v6.csv`, `aldehydes_rejected_v6.csv` |
| `data/raw/screen_v6/` | xTB screen (triage layer, `dG_xtb_kcal`) |
| `data/cross_benzoin/homo_v6/` | **Current featurize OUTPUT** — job 24128375 |
| `data/cross_benzoin/candidates_v2/` | Git-LFS candidate release: 220,859 aldehydes + 1M unordered / 2M directed cross pairs, manifest and QA |
| `data/cross_benzoin/candidates_v3/` | Current Git-LFS candidate release: 220,859 aldehydes + 2M unordered / 4M directed cross pairs, manifest and QA |
| `data/cross_benzoin/cross_pilot_v1/` | **COMPLETE — first real cross-benzoin compute** (job 24607515, ~48 min, 600/600, 0 errors): 300 unordered / 600 directed pairs, aldehyde side 100% cache-hit from `homo_v6/aldehydes_all.csv`; consolidated `cross_pilot_v1_products.csv` (tracked) + `cross_pilot_v1_pairs.csv` (tracked); per-chunk `chunk_NNNN/{products.csv,xyz_prod/}` and `logs/` untracked (gitignored, regenerable from the pairs CSV) |
| ↳ `chunk_NNNN/{products,aldehydes}.csv`, `xyz_prod/`, `xyz_ald/` | Per-chunk results (100 mol/chunk) |
| ↳ `aldehydes_all.csv`, `products_all.csv` | Concatenated tables (created post-array, RUNBOOK step 4) |
| ↳ `logs/cb_<jobid>_<task>.out` | Per-task logs (current job 24128375; defunct 23939656 logs removed) |
| `data/analysis/` | Analysis/figure outputs |

## Models / runs / tracking
| Path | Purpose |
|---|---|
| `runs/models/` | Trained Δ-model artifacts (`delta_model.joblib`, `feature_list.json`, `metadata.json`) |
| `src/benzoin_dG/models/` | Shipped model + `ad_reference.npz` |
| `runs/logs/` | Training/GNN job logs (`train_%j.out`, `dmpnn_%j.out`) |
| `mlflow.db` | MLflow sqlite tracking (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |

## Figures & plotting scripts (register every figure here)
| Script | Output | Purpose |
|---|---|---|
| `pipeline/analysis/library_v6_viz.py` | `data/analysis/library_v6/*.png` | Structural library composition (MW, cho_class, xtb_risk, reject funnel) |
| `pipeline/plot_parity_panels.py` | (per script) | Δ-model parity plots |
| `pipeline/plot_baseline_ab.py` | (per script) | g-xTB vs GFN2 baseline A/B |
| `pipeline/plot_conformer_xtb_dft_conclusions.py` | `figs/concl{1a,1b,2,3}_*_20260622.png` | Conformer xTB–DFT conclusions: 1a geometry swings DFT ΔE (MAD 7.8); 1b g-xTB geom variationally worse (GFN2 wins 97–99%); 2 g-xTB energy≈DFT at fixed geom (MAE 3.3/r0.97 vs GFN2 15/0.57); 3 DFT self-opt 0/13, g-xTB geom→DFT-SP unblocks |
| `figs/NARRATIVE_conformer_xtb_dft_4figs_20260629.md` (中文) + `..._EN.md` (English) | — | Narrative walkthrough of the four concl figures (what/readout/meaning per fig + logic-chain table); notes 1% pilot = uniform random ~1%, and why fixed-geom r0.97 vs full-lib r0.60 don't conflict |
| `pipeline/analysis/library_homo_v6_analysis.py` | `data/analysis/library_homo_v6/<ts>_{dG_xtb_hist,dG_gxtb_hist,gxtb_vs_xtb_hexbin}.png` | 220k library ΔG distributions + method agreement |
| `pipeline/learning_curve_cb.py` | `data/analysis/learning_curve_cb/<ts>_learning_curve.png` | test MAE vs #training labels (label-set sizing) |
| `pipeline/analysis/collect_confnoise.py` | `viz_gxtb_20260625/confnoise_std_hist_<ts>.png` | Single-conformer product-side ΔG std (label-noise upper bound) |
| `pipeline/analysis/collect_boltz_relabel.py` | `viz_gxtb_20260625/boltz_relabel_{corr,func}_hist_<ts>.png` | Boltzmann correction & wB97X-3c functional-shift distributions (better-label probe); report `REPORT_better_labels_20260626.md` |
| `pipeline/plot_cvmae_evolution.py` | `figs/cvmae_evolution_20260629.png` (+ `_nodate_` variant) | cv_mae story 187→funnel_v3: data-plateau ~3.2, D-MPNN above plateau (failed), funnel v1→v3 breakthrough to 2.23 (cleaning>arch≈data); values from mlflow.db benzoin_delta_dG* |
| `pipeline/plot_cvmae_full_journey.py` | `figs/cvmae_full_journey_20260629.png` | FULL arc 06-11→06-29, 2 eras (divider, NOT comparable): Era1 aldehyde-side CV MAE 187→funnel_v3 2.23→g-xTB base 2.00; Era2 cross-benzoin product-side 70/20/10 TEST MAE — ensemble 1.61/arom 1.42, pure-GNN 2.58 loses; mlflow.db + mlflow_benchmark.db |
| `pipeline/plot_screen_v6_models.py` (default / `--no-scaffold` / `--gnn`) | `figs/screen_v6_model_mae_20260629.png`, `_noscaffold_`, `figs/screen_v6_gnn_vs_gbt_20260629.png` | **screen-v6** xTB-ΔG surrogate model MAE (NOT Δ-learning): linear 2.35→MLP/GBT ~2.0 (aromatic) ceiling; +ADCH/QTAIM no gain. Default=full (incl. scaffold Group B); `--no-scaffold`=model-family scan only; `--gnn`=GINE-GNN 2.18 vs GBT 2.14 head-to-head (same scaffold split, +R²/RMSE, 2D-graph loses). From REPORT_screen_v6_models_20260629.md |
| `pipeline/plot_homo_benzoin_journey.py` | `figs/homo_benzoin_gxtb_correction_20260629.png` | **homo-benzoin** g-xTB-ONLY g-xTB→DFT correction benchmark (companion to screen-v6; single metric=70/20/10 TEST MAE): raw g-xTB 4.26→learned correction f34→f56→f72 × Ridge/MLP/XGB/Ensemble → champion Ensemble-f72 1.61 (arom 1.42); ref archs dual-QM-GNN 1.62 / DimeNet++ 2.04 / GINE-hybrid 2.13 / Tier-1 2.75 don't beat it; mlflow_benchmark.db gxtb_dft_full_benchmark. (superseded earlier mixed-metric `homo_benzoin_accuracy_journey_*.png`) |
| `pipeline/plot_scaffold_split_explainer.py` (`--lang en/zh/both`) | `figs/scaffold_split_explainer_en_20260629.png`, `_zh_` | Conceptual schematic of Bemis–Murcko scaffold split vs random split (leakage vs novel-core test); TWO single-language figures (EN + 中文, zh uses DroidSansFallback). Pairs with bilingual `pipeline/docs/EXPLAINER_scaffold_split_20260629.md` |
| _add new plotting scripts + outputs here_ | | |
| `pipeline/analysis/gnn_dual_encoder.py` + `slurm/submit_gnn_dual.sh` | Dual-encoder Δ-GNN (product graph + aldehyde graph, h_P−2·h_A cycle combine); homo: reactant GRAPH redundant (dual 2.54≈product 2.55), value is in reactant QM not topology (job 24212556) |
| `pipeline/analysis/finalize_correction.py` + `slurm/post_dft_chain.sh` | FINALIZE: champion (MLP/GBT on 56 product+reactant feats) on complete DFT labels → save model `pipeline/models/` + full-lib `products_dG_corrected_FINAL`. post_dft_chain auto-fires via --dependency=afterany:24178884 (build retry→submit 7200s retry→finalize) |
| `pipeline/analysis/tier1_smiles_to_dg.py` | TIER-1 triage: pure SMILES→ΔG (Morgan2048+16 RDKit global → XGB, target DFT ΔG direct, NO xTB/QM); test MAE 2.75 / R²0.59; ms/mol; model in pipeline/models/tier1_smiles_to_dG_*.joblib |
| `pipeline/analysis/benchmark_mlflow.py` | UNIFIED full-219k benchmark, all Δ-models same split → MLflow (sqlite:///mlflow_benchmark.db); experiment gxtb_dft_full_benchmark |
| `pipeline/analysis/exp_scaffold_split.py` (job 24224242) | [2] Murcko scaffold-split generalization vs random → MLflow exp2 |
| `pipeline/analysis/exp_quantile_uncertainty.py` (24224243) | [3] XGB quantile PI uncertainty/routing vs ensemble-std → MLflow exp3 |
| `pipeline/analysis/exp_tier1_distill.py` (24224244) | [4] Tier-1 distillation hard-DFT vs soft-teacher target → MLflow exp4 |
| `pipeline/slurm/submit_exp_cpu.sh` | generic genoa CPU submit (SCRIPT=… ) for the exp_* analyses |
| `pipeline/compute/mwf_subset_worker.py` + `analysis/adchqtaim_compare.py` + `slurm/submit_mwf_subset.sh` | ADCH/QTAIM on saved geom (reuses featurize_product funcs); 2.5k subset validation → does NOT help (56-QM 2.44 vs +ADCHQTAIM 2.45); decided NOT to back-fill full-lib |
| `pipeline/analysis/headroom_probe.py` | Tabular headroom (56 feat): deeper MLP/bigger XGB/ensemble + R²; ENSEMBLE MLP+XGB=1.67 R²0.841 (near-saturated) |
| `pipeline/analysis/scope_ensemble_56feat.py` | MLP/XGB/ENSEMBLE × {all,aromatic,aliphatic} on 56 feat; ensemble wins all scopes: all 1.611 / aromatic 1.405 / aliphatic 2.031 |
| `pipeline/analysis/gnn3d_schnet_dimenet.py` + `slurm/submit_gnn3d.sh` | 3D GNNs (SchNet/DimeNet++/equivariant) on product xyz, Δ-learning, 60k; pure-torch radius_graph patch (env lacks torch-cluster); fair same-subset MLP baseline (job 24217134) |
| `pipeline/compute/conformer_noise_worker.py` + `slurm/submit_confnoise.sh` + `analysis/collect_confnoise.py` | LABEL-NOISE FLOOR: K=5 conformers × DFT(r2SCAN-3c) SP per product, per-mol ΔG std/range = irreducible single-conformer label noise bounding the ~1.6 MAE (job 24225782, 36-mol sample; PRODUCT side only). collect_confnoise → results csv + summary md + std hist |
| `pipeline/compute/boltz_relabel_worker.py` + `slurm/submit_boltz_relabel.sh` + `analysis/collect_boltz_relabel.py` | BETTER-LABEL probe (job 24226784, 120 test mols): per mol recompute ΔE=E_prod−2·E_ald 3 ways — r2SCAN-3c single-conf vs **xTB-Boltzmann K=5** (both ald+prod) vs **wB97X-3c** single-conf; form corrected labels, re-score frozen model preds → does multi-conformer / higher-functional re-labeling lower the 1.6 MAE floor? collect → results/summary/2 hists |
