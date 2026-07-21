
# 对比 BDE/BDFE 模型

> 我需要对比 [ElioChen/dG-prediction-of-aldehyde-to-benzoin-products](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products)中的BDE/BDFE预测模型 与ALPHABET以及最新的科研进展进行对比，例如https://doi.org/10.1038/s41467-026-74919-8

<details><summary>4 previous messages</summary>

> 我会先用 GitHub 工作流梳理该仓库的模型、数据与验证方法，再查阅 ALPHABET 的原始论文、你给出的 2026 Nature Communications 论文及近期 BDE/BDFE 研究，最后按“任务定义—数据—表示—精度—泛化—适用边界”做可复核的中文对比。
>
> <details><summary>Ran a command, called 10 tools, searched the web</summary>
>
> - Ran `Get-Content -Raw 'C:\Users\shuai\.codex\plugins\cache\openai-curated-remote\github\0.1.8-2841cf9749ae\skills\github\SKILL.md'`
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `site:github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products ...`
>
> </details>
>
> MCP tool call
>
> codex_apps.github.search_repositories
>
> ```json
> {
>   "query": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "per_page": 5
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "repositories": [
>     {
>       "id": "1299749180",
>       "name": "dG-prediction-of-aldehyde-to-benzoin-products",
>       "owner": {
>         "login": "ElioChen",
>         "id": 47839642,
>         "type": "User",
>         "avatar_url": "https://avatars.githubusercontent.com/u/47839642?v=4",
>         "url": "https://api.github.com/users/ElioChen"
>       },
>       "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>       "permissions": {
>         "admin": true,
>         "maintain": true,
>         "pull": true,
>         "push": true,
>         "triage": true
>       },
>       "allow_auto_merge": null,
>       "allow_merge_commit": null,
>       "allow_rebase_merge": null,
>       "allow_squash_merge": null,
>       "allow_update_branch": null,
>       "clone_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products.git",
>       "size": 148529,
>       "default_branch": "main",
>       "git_commits_url": "https://api.github.com/repos/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/git/commits{/sha}",
>       "git_refs_url": "https://api.github.com/repos/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/git/refs{/sha}",
>       "git_url": "git://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products.git",
>       "master_branch": null,
>       "visibility": "public",
>       "archived": false,
>       "is_code_search_indexed": null,
>       "display_title": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "BDE BDFE",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 20
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "path": "pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bde_surrogate.sh",
>       "path": "pipeline/slurm/submit_bde_surrogate.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bde_surrogate.sh",
>       "display_title": "pipeline/slurm/submit_bde_surrogate.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "path": "pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gxtb_20260707_en.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "path": "pipeline/analysis/train_bde_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "display_title": "pipeline/analysis/train_bde_surrogate.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_pilot.sh",
>       "path": "pipeline/slurm/submit_bdfe_pilot.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_pilot.sh",
>       "display_title": "pipeline/slurm/submit_bdfe_pilot.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde_free_energy.py",
>       "path": "pipeline/compute/calc_bde_free_energy.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde_free_energy.py",
>       "display_title": "pipeline/compute/calc_bde_free_energy.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_array.sh",
>       "path": "pipeline/slurm/submit_bdfe_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_array.sh",
>       "display_title": "pipeline/slurm/submit_bdfe_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_product_bdfe.py",
>       "path": "pipeline/analysis/analyze_product_bdfe.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_product_bdfe.py",
>       "display_title": "pipeline/analysis/analyze_product_bdfe.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_en.md",
>       "path": "pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_en.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_en.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_en.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_zh.md",
>       "path": "pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_zh.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gfn2_raw_energy_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde_free_energy_gxtb.py",
>       "path": "pipeline/compute/calc_bde_free_energy_gxtb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde_free_energy_gxtb.py",
>       "display_title": "pipeline/compute/calc_bde_free_energy_gxtb.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_finalize_bdfe_gxtb_full.sh",
>       "path": "pipeline/slurm/submit_finalize_bdfe_gxtb_full.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_finalize_bdfe_gxtb_full.sh",
>       "display_title": "pipeline/slurm/submit_finalize_bdfe_gxtb_full.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md",
>       "path": "pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_aldehyde_bdfe.py",
>       "path": "pipeline/analysis/analyze_aldehyde_bdfe.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_aldehyde_bdfe.py",
>       "display_title": "pipeline/analysis/analyze_aldehyde_bdfe.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_v2_array.sh",
>       "path": "pipeline/slurm/submit_bdfe_v2_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_v2_array.sh",
>       "display_title": "pipeline/slurm/submit_bdfe_v2_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "path": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_title": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_gxtb_array.sh",
>       "path": "pipeline/slurm/submit_bdfe_gxtb_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bdfe_gxtb_array.sh",
>       "display_title": "pipeline/slurm/submit_bdfe_gxtb_array.sh"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "README.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ````json
> {
>   "content": "# dG-prediction-of-aldehyde-to-benzoin-products\n\nPredict the Gibbs free energy **ΔG** of the NHC-catalyzed **homo-benzoin\ncondensation** (`2 R-CHO → R-CH(OH)-C(=O)-R`) directly from an aldehyde\n**SMILES**, at DFT accuracy but at a fraction of the DFT cost.\n\n```\nΔG_pred = ΔG_g-xTB + model(QM descriptors, ΔG_g-xTB)     # model ≈ ΔG_DFT(r2SCAN-3c) − ΔG_g-xTB\n```\n\nA cheap semi-empirical (g-xTB) estimate of ΔG is corrected up to DFT\n(ORCA **r2SCAN-3c**, CPCM/DMSO) quality by a **Δ-learning** ensemble\n(MLP + gradient-boosted trees + quantile uncertainty), trained on a\nfiltered library of ~220k candidate aldehydes and real DFT single-point\nlabels for nearly the whole library.\n\n## Current best model\n\n**`MORDREDSLIM271_BDEGXTB`** — test MAE **1.503 kcal/mol** (RMSE 2.257, R² 0.875)\non a held-out 70:20:10 split (n≈218k). 275 features: 72 QM/steric descriptors\n(xTB + Multiwfn ADCH/QTAIM, reactant + product) + 199 SHAP/correlation-pruned\nMordred descriptors (dispersion/size/shape) + 4 g-xTB bond-dissociation-energy\nfeatures. Ships with an uncertainty-routing head: the confident 85% of\npredictions hit MAE 1.25; the routed 15% (flagged for DFT follow-up) sit at\nMAE ~2.9.\n\nAromatic substrates predict noticeably better than aliphatic (1.33 vs 1.87\nMAE) — historical sampling bias toward aromatics has been confirmed resolved\nat full-library scale. The dominant remaining error driver is electronic\n(sulfonyl/phosphorus/imine substituents), not geometry or missing descriptors.\n\n## What we tried and why it isn't the production model\n\n- **GNN (graph neural net) on the same 275 features**: a dual-encoder GINE\n  architecture (product graph + aldehyde graph, thermodynamic-cycle combine)\n  given the exact same feature set as the tabular champion still loses\n  standalone (MAE 1.55 vs 1.50) — architecture is not the lever once the\n  information content is held fixed. GNN+tabular *stacking* showed a\n  promising signal in earlier partial-overlap checks; a full-library,\n  leak-free re-verification is in progress before any promotion claim.\n- **Pure-SMILES / no-QM-descriptor baselines** (SELFIES bag-of-symbols,\n  ECFP fingerprints, a from-scratch sequence model, a from-scratch GINE\n  graph model, and a fine-tuned pretrained ChemBERTa): all land in the\n  2.7–3.3 kcal/mol MAE range on this task even at full library scale —\n  well short of the descriptor-informed Δ-learning model. The QM/xTB layer\n  is where most of the achievable accuracy lives, not the architecture\n  consuming it.\n\n## Install\n\n```bash\npip install -e .                 # library + inference\npip install -e \".[pipeline]\"     # + data-gen / training (MLflow, Optuna, …)\n```\n\nExternal binaries (not pip deps) are discovered at runtime:\n\n```bash\nexport XTB_BIN=/path/to/xtb                    # required for inference\nexport MULTIWFN_BIN=/path/to/Multiwfn_noGUI     # optional (ADCH/QTAIM features)\n```\n\nWithout Multiwfn the ADCH/QTAIM descriptors fall back to training medians.\n\n## Use\n\n```bash\nbenzoin-dg \"O=Cc1ccccc1\"                 # benzaldehyde\nbenzoin-dg \"O=Cc1ccccc1\" \"O=CCC\" --json\n```\n\n```python\nfrom benzoin_dG import predict_dG\np = predict_dG(\"O=Cc1ccccc1\")\nprint(p.dG_pred, p.dG_xtb, p.dG_correction)\n```\n\n## Layout\n\n```\nsrc/benzoin_dG/         installable library (import benzoin_dG)\n  descriptors.py        1-molecule featurization (wraps _descriptors_backend)\n  thermo.py             xTB ΔG (wraps _thermo_backend, ORCA off)\n  features.py           assemble + median-impute the model input vector\n  predict.py / cli.py   SMILES -> ΔG\n  models/                shipped trained model + feature spec\npipeline/                research / data-generation pipeline\n  filter_smiles_v6.py    current production SMILES filter (raw -> clean library)\n  compute/               xTB/g-xTB/DFT featurization + descriptor workers\n  analysis/              model training, finalize_*, GNN experiments, SHAP, diagnostics\n  slurm/                 HPC submit scripts for the above\n  docs/                  methods notes, weekly reports, data-split docs\ncross_benzoin/           cross-substrate (non-homo) benzoin extension\ndata/library/             filtered aldehyde libraries (v2-v6) + rejects\ndata/cross_benzoin/       featurization + model outputs for the homo-benzoin study\ndata/analysis/             plots/reports from various analysis passes\ntests/\n```\n\nSee [FILE_MAP.md](FILE_MAP.md) for a more complete file-by-file index and\n[ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline architecture.\n\n## Status (2026-07-13)\n\nProduction champion (`MORDREDSLIM271_BDEGXTB`, MAE 1.503) is trained and has\nscored the full ~220k-molecule filtered library. GNN and pure-SMILES\nalternatives have been explored extensively (see above) and, for this task,\ndescriptor-informed Δ-learning on gradient-boosted trees remains the\nstrongest approach. Active work: verifying whether GNN+tabular stacking adds\na real, reproducible gain at full library scale.\n\n**Note on git history:** this repository's local `.git` object database was\nfound corrupted (silent object-store corruption, not user error) while\npreparing this upload; history was restarted from the current, verified-intact\nworking tree on 2026-07-13 rather than risk pushing corrupted objects.\n",
>   "encoding": "utf-8",
>   "sha": "bda6b0748a1c889abc3da9ea1e4f705fcb22294c",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/README.md",
>   "display_title": "README.md"
> }
> ````
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# BDE/BDFE 方法层级 2：GFN2-xTB 自由能校正 BDFE\n\n**状态：空结果——未采纳**\n\n## 动机\n\n原始电子能 BDE（层级 1）忽略了零点能、热力学和熵校正。BDFE（键解离**自由能**）是热力学上更\n严谨的量，预期能更好地关联反应的真实能量学——尤其是醛 C-H 键（直接关系到 Breslow 中间体形成\n的热力学）和产物 C-C 键（真正生成的那根键）。\n\n## 方法\n\n两版实现，v1 到 v2（v2 修复了 v1 的一个溶剂一致性 bug）：\n\n- **v1**（`pipeline/compute/calc_bde_free_energy.py`）：对母体和每个自由基片段做 xtb\n  `--ohess`（完整数值 Hessian + RRHO 热校正），**气相**（无隐式溶剂），再用 Shermo 对\n  g98.out 做 qRRHO 重校正（复用 `thermo_orca.py` 的 `run_ohess`/`run_shermo`/`parse_shermo_G`\n  模式）。孤立 H 原子片段用解析的理想单原子气体校正（Sackur-Tetrode + 电子双重简并——单原子\n  无法做振动 Hessian）。\n- **v2**（`pipeline/compute/calc_bde_free_energy_v2.py`）：物理原理相同，但 (a) 复用已有\n  funnel_v3 特征化中母体的 G/E，无需重算（只有两个新的自由基片段需要重新做 `--ohess`，xtb\n  成本降低约一半到三分之二），(b) 在所有片段计算上加 `--alpb dmso`。**v1 从头到尾都是气相\n  （包括母体）**——把溶剂化的母体数值和气相片段混用（或者说原母体重算版本本身处处气相），相对于\n  项目其余部分统一采用 DMSO 溶剂化的方法论而言是一个真实的不一致，不只是舍入误差。重新验证后：\n  片段改为 DMSO 一致后，与 v1 的差距从 4.6 kcal/mol 降到一个符合物理预期的约 1.7 kcal/mol 残余\n  差（闭壳层母体与开壳层自由基片段之间的差异性溶剂化稳定作用——这是真实效应，不是 bug）。\n\n## 中试验证\n\n20 分子中试（10 个醛 + 10 个产物），成功率 100%：经 G 校正后的 BDE 比原始电子能**低约 14-19\nkcal/mol**（醛 C-H：E 102.8-116.7 → G_xtb 89.0-102.3，低端已接近约 88 kcal/mol 的实验参考值；\n产物 C-C：E 75-107 → G_xtb 58-90）。Shermo 与 xtb 自身的 RRHO 结果相差不到 0.5 kcal/mol——\n该样本中没有严重的低频振动模式病理性问题。\n\n过程中发现并修复了一个 bug：`_frag_G` 最初对所有物种（包括闭壳层母体）都硬编码\n`--uhf 1`（母体应为 `uhf=0`），导致 xtb 立即报致命错误；修复方式是给每个物种传入独立的 uhf\n参数。\n\n## 覆盖率\n\n醛（v2，任务 24416982）：1459/1471（99.2%，接受 12 个超时）。产物（v2，任务 24422675，此前\n因 60min/chunk 出现 28% 超时而改为 120min/chunk 重新提交）：1435/1463（98.1%，接受 28 个\n超时）。附属文件：`aldehydes_bdfe2_descriptors.csv`（218,724 行，填充率 97.9%）、\n`products_bdfe2_descriptors.csv`（215,367 行，填充率 96.0%）。\n\n## 结果\n\n`finalize_correction_bdfe_full.py`（任务 24434658），与层级 1 相同的快速检验流程：\n\n| 配置 | 特征数 | 测试集 MAE | 差值 |\n|---|---|---|---|\n| mordredslim271 基线 | 271 | 1.612 | — |\n| + BDFE（GFN2，两侧） | 273 | 1.605 | **-0.007（空结果）** |\n\n差值远低于 0.02-0.03 的噪声带，且明显弱于层级 1 原始电子能 BDE 的提升（-0.024）。仅醛侧的\n快速检验（任务 24427785）单独也是同样的结论：+0.003，基本没有信号。\n\n## 解读\n\n尽管 BDFE 在热力学上是更严谨的量，它在这个 ML 任务上却**没有**超越更粗糙的原始电子能。可能的\n解释是：`--ohess` 带来的额外热/熵校正项（尤其是低频振动模式，众所周知对模式选择敏感、数值上\n噪声较大）引入的噪声可能和物理信号一样多，而原始电子能虽\"不那么严谨\"，却是一个更干净的\n逐分子量，恰好更符合校正模型的需要。\n\n**决定：不采纳 GFN2 层级的 BDFE。** 这一结论在 g-xTB 层级（层级 3）再次成立——BDFE 单独使用\n在那里依然是空结果，证实这个模式源于 RRHO 校正本身，而非底层电子结构方法。\n",
>   "encoding": "utf-8",
>   "sha": "5155be3d6772928676a23e6cd7089f6c4a96c037",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/docs/METHODS_BDE_gfn2_free_energy_20260707_zh.md",
>   "display_title": "METHODS_BDE_gfn2_free_energy_20260707_zh.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# BDE/BDFE surrogate model (20260704)\n\nPredicts per-molecule bond dissociation (free) energy from the existing cheap descriptors (72-champion QM + mordredslim271 kept mordred subset), instead of running real xtb (--ohess for BDFE). Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble + quantile(05/95) UQ, per target.\n\n## Results\n\n| target | n | n_feat | test MAE | RMSE | R2 |\n|---|---|---|---|---|---|\n| bde_ald_CH_kcal (ald_bde) | 214,954 | 140 | 3.027 | 9.700 | 0.094 |\n| bdfe_xtb_kcal (ald_bdfe) | 213,191 | 140 | 2.018 | 4.436 | 0.797 |\n| bde_prod_CC_kcal (prod_bde) | 208,907 | 147 | 3.878 | 10.334 | 0.652 |\n| bdfe_xtb_kcal (prod_bdfe) | 205,339 | 147 | 2.704 | 5.117 | 0.901 |\n\n## Interpretation\n\n- **ald_bde**: R2=0.094 -> redundancy is **LOW (carries genuinely independent information)**. Top predictive feats: mordred_Mor17, g_nAliphRing, mordred_Mor07m, mordred_Mor23p, mordred_Mor20v\n- **ald_bdfe**: R2=0.797 -> redundancy is **MODERATE**. Top predictive feats: xtb_LUMO, wbo_CO, fukui_plus_CHO_C, xtb_EA, dual_descriptor_CHO_C\n- **prod_bde**: R2=0.652 -> redundancy is **MODERATE**. Top predictive feats: mordred_MOMI-Y, SASA_total, mordred_PBF, sterimol_B5, mordred_Mor30m\n- **prod_bdfe**: R2=0.901 -> redundancy is **HIGH (mostly redundant with existing feats)**. Top predictive feats: wbo_CC_new, wbo_CO_ket, xtb_LUMO, hb_angle, g_nAromRing\n\nIf R2 is high enough for practical use (rule of thumb R2>0.85, MAE well under the real xtb run's own noise), the corresponding bundle can be used as a fast prospective-screening substitute for real xtb BDE/BDFE on new molecules outside the current 220k library.\n\n## Model bundles\n\n- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_ald_bde_20260704.joblib`\n- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_ald_bdfe_20260704.joblib`\n- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_prod_bde_20260704.joblib`\n- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_prod_bdfe_20260704.joblib`\n",
>   "encoding": "utf-8",
>   "sha": "d9de158abb6513cb3b9feb87d4bcba3ceb5c9f22",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md",
>   "display_title": "REPORT_bde_surrogate_20260704.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# BDE/BDFE 方法层级 3：g-xTB 一致的 BDE + BDFE\n\n**状态：已采纳——当前生产冠军特征集**\n\n## 动机（\"方法不匹配\"假设）\n\n层级 1-2 都在 GFN2-xTB 层级计算 BDE/BDFE，但实际被校正的生产基线（dG_gxtb）是\n**g-xTB**，不是 GFN2——GFN2 只是内部混合公式中提供热校正的辅助量（见 `gxtb_baseline.py` 的\n`G_gxtb = E_gxtb + (G_gfn2 - E_gfn2)`）。假设：描述符（GFN2）与它要帮助校正的对象\n（g-xTB 自身的误差）之间的层级不匹配，可能稀释了信号——如果这个假设成立，在 g-xTB 层级重新\n计算 BDE/BDFE 应该能携带更多信号。\n\n## 方法\n\n`pipeline/compute/calc_bde_free_energy_gxtb.py`，复用项目自己的混合校正模式：\n- 母体的 `G_gxtb` 已经缓存（`{aldehydes,products}_all.csv`）——无需重新计算。\n- 母体的原始 `E_gxtb` 可以**免费代数求出**：`E_gxtb_母体 = G_gxtb_母体 - (G_gfn2_母体 -\n  E_gfn2_母体)`（对混合公式取逆；两个 GFN2 项都是已缓存的 `G_xtb`/`xtb_energy` 列）。\n- 每个自由基片段做一次全新的 GFN2 `--ohess`（热校正 + 弛豫几何，同层级 2），**外加在同一\n  几何上做一次 g-xTB 单点**（`--gxtb --cosmo dmso`，已冒烟测试可同时使用——g-xTB 的解析\n  Hessian 可能比 GFN2 的数值 Hessian 更数值稳定，这与层级 2 中\"低频模式噪声\"的猜测相关）。\n  `bde_gxtb_kcal` 和 `bdfe_gxtb_kcal` 都由同一次片段计算同时产出——片段侧的 `E_gxtb` 本来\n  就是单点步骤内部算出的，此前被丢弃，现在直接保留。\n\n## 中试验证\n\n3 分子中试：3/3 成功，`BDE > BDFE`，符合解离熵效应的预期。分散采样中试（而非仅取前 N\n行——那些全是小的脂肪族分子，耗时高度依赖分子大小，每 20 分子 chunk 从 30 秒到 14 分钟不等）：\n- n≈140/侧：醛 C-H BDFE(g-xTB) 与 dG_gxtb 的相关性 r=0.300（p=3e-4）——相对 GFN2 版本\n  在全库上的 r=0.044 是真实的跃升。产物 C-C BDFE(g-xTB) r=-0.184，与 GFN2 版本量级相近——\n  产物侧没有改善。\n- n≈711 醛 / 694 产物（更大的合并中试）：醛侧 r=0.209（p=1.9e-8），仍是 GFN2 版本 r=0.044\n  的约 4-5 倍，且高度显著；产物侧 r=-0.254，相比 GFN2 的 -0.173 仍无明显改善。\n\n## 覆盖率\n\n全库阵列：醛 1460/1471（99.3%，任务 24437300，干净完成）；产物 1459/1463（99.7%，任务\n24437301→24454920，中间因共享环境损坏事故被迫回填——见 `shared-env-instability-2026-07-05`\n——通过隔离的 `envs/bde_lite` venv 恢复）。附属文件：\n`{aldehydes,products}_bdfe_gxtb_descriptors.csv`，共 219,095 行有标签数据，各列覆盖率\n95.8-97.8%（仅在训练集上做中位数填补）。\n\n## 结果\n\n`finalize_correction_bdfe_gxtb_full.py` 在 mordredslim271（271 特征，MAE 1.612）上做快速\n2 成员 XGB 检验：\n\n| 配置 | 特征数 | 测试集 MAE | 差值 |\n|---|---|---|---|\n| mordredslim271 基线 | 271 | 1.612 | — |\n| + BDFE(g-xTB)，两侧 | 273 | 1.605 | -0.007（空结果，与 GFN2 自身的空结果一致） |\n| + BDE(g-xTB)，两侧 | 273 | **1.580** | **-0.032（真实——大于 GFN2 自身的 +0.024）** |\n| + BDE(g-xTB) + BDFE(g-xTB)，两侧 | 275 | **1.563** | **-0.049（迄今最佳结果，约为噪声带的 4 倍）** |\n\n**完整生产确认**（`finalize_correction_mordredslim271_bdegxtb.py`，任务 24468737，真正的\nMLP + XGB_d8 + XGB_d10 + 分位数不确定性集成，而非快速 2 成员检验）：**测试集 MAE 1.503**，\n相对 mordredslim271 的 1.525 差值为 -0.022，正好卡在噪声带边缘（完整集成本身已经吸收了部分\n这些原始特征在纯 2-XGB 检验上带来的增益，所以增益不会 1:1 从快速检验转移过来）。仍判定为\n真实且值得采纳的提升。\n\n## 解读\n\n方法不匹配假设得到证实，但**仅对 BDE（原始电子能）成立，对 BDFE 不成立**。g-xTB 一致的\nBDE 超越了 GFN2 层级 BDE 本已勉强的提升，而 BDE+BDFE 合并使用是超出噪声带的明确胜利——尽管\nBDFE **单独**使用依然是空结果，与 GFN2 的模式完全一致。BDFE 中额外的 RRHO 热校正项看起来\n**无论底层电子结构方法是什么**都更像噪声而非信号，但当 BDE 在与被校正对象真正匹配的层级上\n计算时，其原始电子能会携带更多信息。\n\n## 成本感知后续分析（2026-07-07 SHAP 审计）\n\n对完整 275 特征冠军模型做 SHAP 重要性分析（4000 行测试子样本，XGB_d8）：`ald_bde_gxtb_kcal`\n排名 4/275（mean|SHAP|=0.587），`prod_bde_gxtb_kcal` 排名 6/275（0.484)——两者都**便宜**\n（单次 SP/opt）。`prod_bdfe_gxtb_kcal` 排名 15/275（0.191），`ald_bdfe_gxtb_kcal` 排名\n38/275（0.099）——两者都**昂贵**（每个片段都需要完整的 `--ohess` Hessian+RRHO）。重要性\n求和：BDE=1.070，BDFE=0.290（比值 3.7 倍）。**对未来新分子的前瞻性筛选建议：只计算 BDE，\n跳过 BDFE**——重要性损失接近零，却能省下大量计算（无需 Hessian）。现有生产 bundle 保留两者\n（全库已经算过，成本已沉没），但这应指导未来的特征工程方向：不再投入 BDFE 族（热/熵）描述符。\n\n另见 [METHODS_BDE_gfn2_raw_energy_20260707_zh.md](METHODS_BDE_gfn2_raw_energy_20260707_zh.md)\n和 [METHODS_BDE_gfn2_free_energy_20260707_zh.md](METHODS_BDE_gfn2_free_energy_20260707_zh.md)\n——本层级取代的两个更早方法层级。\n",
>   "encoding": "utf-8",
>   "sha": "0b5fbba32ae4b106f8e933eabd858f8d3db26192",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>   "display_title": "METHODS_BDE_gxtb_20260707_zh.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"Full go/no-go for the g-xTB-CONSISTENT BDFE/BDE descriptors (method-mismatch hypothesis,\nsee bde-descriptor-idea memory): the GFN2-level BDFE was a NULL RESULT\n(finalize_correction_bdfe_full.py: +0.007 MAE, noise-level) because the descriptor's level\nof theory (GFN2) didn't match the thing it's meant to help correct (g-xTB's own errors).\ncalc_bde_free_energy_gxtb.py recomputes BOTH bdfe_gxtb_kcal (free-energy-corrected, matches\nthis project's G_gxtb hybrid-correction pattern) and bde_gxtb_kcal (raw electronic energy,\nfree byproduct of the same calc -- see bde-descriptor-idea's \"also added BDE_gxtb for free\"\nnote) for the aldehyde C(=O)-H bond and the product's new C-C bond, now g-xTB-consistent.\n\nFull-library arrays now complete on both sides (aldehydes job 24460460: 1471/1471, 100%;\nproducts backfill 24454920 + original: 1460/1463, 3 timeout gap accepted) -- this is the\nreal verdict on the method-mismatch hypothesis, mirroring finalize_correction_bdfe_full.py's\nexact protocol (same 70:20:10 split, same quick XGB_d8+XGB_d10 ensemble, same mordredslim271\nbaseline) but swapping in the g-xTB-consistent sidecars. Tests 4 configs in one run since\nboth bdfe_gxtb_kcal and bde_gxtb_kcal come from the SAME calc at zero extra compute cost:\n  1. mordredslim271                          (baseline, test MAE 1.525 production champion)\n  2. + BDFE(gxtb, both sides)                (the pilot's promising aldehyde-side correlation)\n  3. + BDE(gxtb, both sides)                 (raw-E; GFN2's raw-E BDE had the only real gain, +0.024)\n  4. + BDFE(gxtb) + BDE(gxtb), both sides    (both descriptors, in case they're complementary)\n\"\"\"\nimport json, time\nfrom pathlib import Path\nimport numpy as np, pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom xgboost import XGBRegressor\nfrom rdkit import Chem, RDLogger\nfrom rdkit.Chem import rdMolDescriptors, Descriptors\nRDLogger.DisableLog('rdApp.*')\n\nR = \"/scratch-shared/schen3/benzoin-dg\"; H = f\"{R}/data/cross_benzoin/homo_v6\"\nOUT = Path(f\"{H}/viz_gxtb_20260625\"); OUT.mkdir(exist_ok=True)\nTAG = time.strftime(\"%Y%m%d\")\n\nPROD_QM = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\n  \"xtb_dipole\",\"mulliken_ketC\",\"mulliken_ketO\",\"mulliken_carbC\",\"mulliken_hydO\",\"mulliken_hydH\",\n  \"wbo_CO_ket\",\"wbo_CC_new\",\"wbo_CO_carb\",\"fukui_plus_ketC\",\"fukui_minus_ketC\",\"dual_ketC\",\n  \"fukui_plus_carbC\",\"fukui_minus_carbC\",\"dual_carbC\",\"vbur_ketC\",\"vbur_carbC\",\"sterimol_L\",\n  \"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\",\"pa_ketO\",\"hb_dist\",\"hb_angle\",\"dih_core\"]\nALD = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\"xtb_dipole\",\n  \"mulliken_CHO_C\",\"mulliken_CHO_O\",\"fukui_plus_CHO_C\",\"fukui_minus_CHO_C\",\"dual_descriptor_CHO_C\",\n  \"wbo_CO\",\"pa_CHO_O\",\"vbur_CHO_C\",\"sterimol_L\",\"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\"]\nALDp = [f\"ald_{c}\" for c in ALD]\nGKEYS = [\"TPSA\",\"HBD\",\"HBA\",\"RotB\",\"FracCsp3\",\"nHetero\",\"MolWt\",\"nRing\",\"nAromRing\",\"nAliphRing\",\n         \"nAmide\",\"has_P\",\"has_B\",\"has_S\",\"has_Si\",\"has_halogen\"]\nGLOB = [f\"g_{k}\" for k in GKEYS]\nFEATS_72 = PROD_QM + ALDp + GLOB\n\n\ndef gfeats(smi):\n    m = Chem.MolFromSmiles(str(smi))\n    if m is None: return {f\"g_{k}\": np.nan for k in GKEYS}\n    s = {a.GetSymbol() for a in m.GetAtoms()}\n    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),\n            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),\n            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),\n            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),\n            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),\n            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]\n    return {f\"g_{k}\": v for k, v in zip(GKEYS, vals)}\n\n\ndef add_global(df, smi_col):\n    u = df[[smi_col]].drop_duplicates()\n    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values\n    return df.merge(g, on=smi_col, how=\"left\")\n\n\ndef _xgb(depth, ne):\n    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,\n                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,\n                        early_stopping_rounds=60, eval_metric=\"mae\")\n\n\ndef run(df, feats, label):\n    rng = np.random.default_rng(42); idx = rng.permutation(len(df))\n    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]\n    imp = SimpleImputer(strategy=\"median\").fit(df[feats].values[tr])\n    Xtr, Xva, Xte = imp.transform(df[feats].values[tr]), imp.transform(df[feats].values[va]), imp.transform(df[feats].values[te])\n    sc = StandardScaler().fit(Xtr)\n    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)\n    dtr, dva = df.delta.values[tr], df.delta.values[va]; gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]\n\n    members = [(\"XGB_d8\", _xgb(8, 1500)), (\"XGB_d10\", _xgb(10, 2000))]\n    preds_te = []\n    for nm, m in members:\n        m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)\n        preds_te.append(m.predict(Xte))\n    pred = np.vstack(preds_te).mean(0)\n    yhat = gte + pred; err = np.abs(yhat - yte)\n    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))\n    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())\n    scope = {}\n    for s in [\"aromatic\", \"aliphatic\"]:\n        mk = df.cls.values[te] == s\n        if mk.sum() > 50: scope[s] = float(err[mk].mean())\n    print(f\"[{label}] n_feat={len(feats)} n={len(df):,} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} scope={scope}\", flush=True)\n    return {\"label\": label, \"n_feat\": len(feats), \"n\": len(df), \"mae\": mae, \"rmse\": rmse, \"r2\": r2, \"scope\": scope}\n\n\ndef main():\n    cons = Path(f\"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet\")\n    dft = pd.read_parquet(cons, columns=[\"id\", \"dG_orca_kcal\"]).dropna(subset=[\"dG_orca_kcal\"]).drop_duplicates(\"id\", keep=\"last\")\n\n    p = pd.read_csv(f\"{H}/products_all.csv\", usecols=[\"id\", \"donor_id\", \"smiles\", \"dG_gxtb_kcal\"] + PROD_QM, low_memory=False)\n    a = pd.read_csv(f\"{H}/aldehydes_all.csv\", usecols=[\"id\"] + ALD, low_memory=False).drop_duplicates(\"id\").rename(columns={\"id\": \"ald_id\", **{c: f\"ald_{c}\" for c in ALD}})\n    cls = pd.read_parquet(f\"{H}/aldehyde_class.parquet\")\n\n    kept_mordred = set(json.load(open(f\"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json\"))[\"kept_mordred\"])\n    prod_kept = [c for c in kept_mordred if not c.startswith(\"ald_\")]\n    ald_kept_raw = [c[len(\"ald_\"):] for c in kept_mordred if c.startswith(\"ald_\")]\n\n    prod_header = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", nrows=0).columns\n    prod_want = [\"id\"] + [c for c in prod_header if c in prod_kept]\n    prod_mrd = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", usecols=prod_want, low_memory=False)\n    ald_header = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", nrows=0).columns\n    ald_want = [\"id\"] + [c for c in ald_header if c in ald_kept_raw]\n    ald_mrd = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", usecols=ald_want, low_memory=False)\n    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith(\"mordred_\")]\n    ald_mrd = ald_mrd.rename(columns={\"id\": \"ald_id\"})\n    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith(\"mordred_\")]\n    ald_mrd = ald_mrd.rename(columns={c: f\"ald_{c}\" for c in ald_mrd_raw})\n    ald_mrd_cols = [f\"ald_{c}\" for c in ald_mrd_raw]\n    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors=\"coerce\")\n    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors=\"coerce\")\n\n    # g-xTB-CONSISTENT BDFE/BDE sidecars (both sides), freshly concatenated from the now-\n    # complete arrays (aldehydes 1471/1471, products 1460/1463).\n    prod_bdfe = pd.read_csv(f\"{H}/products_bdfe_gxtb_descriptors.csv\",\n                             usecols=[\"id\", \"bdfe_gxtb_kcal\", \"bde_gxtb_kcal\"]).rename(\n        columns={\"bdfe_gxtb_kcal\": \"prod_bdfe_gxtb_kcal\", \"bde_gxtb_kcal\": \"prod_bde_gxtb_kcal\"})\n    ald_bdfe = pd.read_csv(f\"{H}/aldehydes_bdfe_gxtb_descriptors.csv\",\n                            usecols=[\"id\", \"bdfe_gxtb_kcal\", \"bde_gxtb_kcal\"]).rename(\n        columns={\"id\": \"ald_id\", \"bdfe_gxtb_kcal\": \"ald_bdfe_gxtb_kcal\", \"bde_gxtb_kcal\": \"ald_bde_gxtb_kcal\"})\n    # sanity filter: drop pathological non-converged-SCF rows (|value|>200 kcal/mol impossible\n    # for a real bond), same threshold as the GFN2 version.\n    for c in [\"prod_bdfe_gxtb_kcal\", \"prod_bde_gxtb_kcal\"]:\n        prod_bdfe.loc[prod_bdfe[c].abs() > 200, c] = np.nan\n    for c in [\"ald_bdfe_gxtb_kcal\", \"ald_bde_gxtb_kcal\"]:\n        ald_bdfe.loc[ald_bdfe[c].abs() > 200, c] = np.nan\n\n    full = p.copy(); full[\"ald_id\"] = full[\"donor_id\"].astype(\"Int64\"); full = full.merge(a, on=\"ald_id\", how=\"left\")\n    full = add_global(full, \"smiles\")\n    full = full.merge(prod_mrd[[\"id\"] + prod_mrd_cols], on=\"id\", how=\"left\")\n    full = full.merge(ald_mrd[[\"ald_id\"] + ald_mrd_cols], on=\"ald_id\", how=\"left\")\n    full = full.merge(prod_bdfe, on=\"id\", how=\"left\")\n    full = full.merge(ald_bdfe, on=\"ald_id\", how=\"left\")\n\n    df = full.merge(dft, on=\"id\").merge(cls, on=\"id\", how=\"left\")\n    df = df.dropna(subset=[\"dG_gxtb_kcal\", \"dG_orca_kcal\"] + FEATS_72).reset_index(drop=True)\n    df = df[df[\"dG_orca_kcal\"].abs() < 60].reset_index(drop=True)\n    df[\"delta\"] = df[\"dG_orca_kcal\"] - df[\"dG_gxtb_kcal\"]\n    print(f\"labeled rows: {len(df):,}\", flush=True)\n    print(f\"g-xTB BDFE coverage: prod {df['prod_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%  \"\n         f\"ald {df['ald_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%\", flush=True)\n    print(f\"g-xTB BDE coverage:  prod {df['prod_bde_gxtb_kcal'].notna().mean()*100:.1f}%  \"\n         f\"ald {df['ald_bde_gxtb_kcal'].notna().mean()*100:.1f}%\", flush=True)\n\n    feats_slim271 = FEATS_72 + prod_mrd_cols + ald_mrd_cols\n    feats_bdfe = feats_slim271 + [\"prod_bdfe_gxtb_kcal\", \"ald_bdfe_gxtb_kcal\"]\n    feats_bde = feats_slim271 + [\"prod_bde_gxtb_kcal\", \"ald_bde_gxtb_kcal\"]\n    feats_both = feats_slim271 + [\"prod_bdfe_gxtb_kcal\", \"ald_bdfe_gxtb_kcal\",\n                                   \"prod_bde_gxtb_kcal\", \"ald_bde_gxtb_kcal\"]\n    print(f\"mordredslim271: {len(feats_slim271)} feats; \"\n          f\"+BDFE(gxtb): {len(feats_bdfe)}; +BDE(gxtb): {len(feats_bde)}; +both: {len(feats_both)}\", flush=True)\n\n    results = {}\n    configs = [\n        (\"mordredslim271\", feats_slim271),\n        (\"mordredslim271_plus_bdfe_gxtb_both\", feats_bdfe),\n        (\"mordredslim271_plus_bde_gxtb_both\", feats_bde),\n        (\"mordredslim271_plus_bdfe_and_bde_gxtb_both\", feats_both),\n    ]\n    for label, feats in configs:\n        results[label] = run(df, feats, label)\n\n    rep = OUT / f\"REPORT_bdfe_gxtb_full_augment_{TAG}.md\"\n    with open(rep, \"w\") as fh:\n        fh.write(f\"# g-xTB-consistent BDFE/BDE augmented feature comparison ({TAG})\\n\\n\")\n        fh.write(\"Method-mismatch hypothesis test: GFN2-level BDFE was a null result \"\n                 \"(finalize_correction_bdfe_full.py, +0.007 MAE). This uses g-xTB-consistent \"\n                 \"BDFE/BDE (calc_bde_free_energy_gxtb.py), full library both sides \"\n                 \"(aldehydes 1471/1471, products 1460/1463, 3 timeout gap accepted). \"\n                 \"Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble, vs mordredslim271 \"\n                 \"preferred production model (test MAE 1.525).\\n\\n\")\n        for label, r in results.items():\n            fh.write(f\"- **{label}**: n_feat={r['n_feat']} n={r['n']:,} \"\n                     f\"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.3f} scope={r['scope']}\\n\")\n    json.dump(results, open(OUT / f\"bdfe_gxtb_full_augment_results_{TAG}.json\", \"w\"), indent=2)\n    print(\"wrote\", rep, flush=True)\n    print(\"DONE\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "349baf2d4d1598d3f290a343e9e8c4683ffdeea7",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>   "display_title": "finalize_correction_bdfe_gxtb_full.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"Pure-SMILES (no-xtb) surrogate for product C-C BDFE -- the fast/no-quantum tier, mirroring\nthis project's established 2D-surrogate pattern (pipeline/train_surrogate.py, which does the\nsame for the main reaction dG). The QM-feature surrogate (train_bde_surrogate.py) already\ngets R2=0.901 for product BDFE using 72-champion QM + mordred feats -- but those need an xtb\ngeometry+SP to obtain, so they aren't \"free\" for screening brand-new molecules. This script\ntests how much of that predictive power survives using ONLY RDKit 2D descriptors computable\ndirectly from the SMILES string (no conformer, no xtb at all) -- the real no-quantum-cost\ncomparison point.\n\nUses the GFN2-level product BDFE (products_bdfe2_descriptors.csv, full library, 96% filled)\nas the training target since it's the largest available label set right now.\n\"\"\"\nimport sys\nimport time\nfrom pathlib import Path\nimport numpy as np, pandas as pd, joblib\nimport matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom xgboost import XGBRegressor\n\nsys.path.insert(0, str(Path(__file__).resolve().parent.parent / \"compute\"))\nfrom ald_descriptors import calc_rdkit, _RDKIT_FIELDS\n\nR = \"/scratch-shared/schen3/benzoin-dg\"; H = f\"{R}/data/cross_benzoin/homo_v6\"\nOUT = Path(f\"{H}/viz_gxtb_20260625\"); OUT.mkdir(exist_ok=True)\nTAG = time.strftime(\"%Y%m%d\"); MODELDIR = Path(f\"{R}/pipeline/models\"); MODELDIR.mkdir(exist_ok=True)\n\n\ndef _xgb(depth, ne):\n    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,\n                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,\n                        early_stopping_rounds=60, eval_metric=\"mae\")\n\n\ndef savefig(name):\n    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches=\"tight\"); plt.close()\n    print(\"wrote\", name, flush=True)\n\n\ndef main():\n    p = pd.read_csv(f\"{H}/products_all.csv\", usecols=[\"id\", \"smiles\"], low_memory=False).drop_duplicates(\"id\")\n    bdfe = pd.read_csv(f\"{H}/products_bdfe2_descriptors.csv\", usecols=[\"id\", \"bdfe_xtb_kcal\"])\n    bdfe = bdfe[bdfe[\"bdfe_xtb_kcal\"].abs() <= 200]  # drop pathological SCF-failure rows\n\n    df = p.merge(bdfe, on=\"id\", how=\"inner\")\n    print(f\"n={len(df):,} products with valid BDFE label\", flush=True)\n\n    print(\"computing pure-SMILES 2D descriptors (no xtb)...\", flush=True)\n    t0 = time.time()\n    feat_rows = [calc_rdkit(s) for s in df[\"smiles\"]]\n    print(f\"done in {time.time()-t0:.1f}s\", flush=True)\n    feat_df = pd.DataFrame(feat_rows)\n    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)\n    feats = _RDKIT_FIELDS\n    for c in feats:\n        df[c] = pd.to_numeric(df[c], errors=\"coerce\")\n\n    d = df.dropna(subset=feats + [\"bdfe_xtb_kcal\"]).reset_index(drop=True)\n    print(f\"n={len(d):,} after dropna, n_feat={len(feats)}\", flush=True)\n\n    rng = np.random.default_rng(42); idx = rng.permutation(len(d))\n    ntr, nva = int(.7 * len(d)), int(.9 * len(d)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]\n    imp = SimpleImputer(strategy=\"median\").fit(d[feats].values[tr])\n    Xtr_raw, Xva_raw, Xte_raw = imp.transform(d[feats].values[tr]), imp.transform(d[feats].values[va]), imp.transform(d[feats].values[te])\n    sc = StandardScaler().fit(Xtr_raw)\n    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)\n    ytr, yva, yte = d[\"bdfe_xtb_kcal\"].values[tr], d[\"bdfe_xtb_kcal\"].values[va], d[\"bdfe_xtb_kcal\"].values[te]\n\n    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)\n    xgb8.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    xgb10.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    yhat = np.vstack([xgb8.predict(Xte), xgb10.predict(Xte)]).mean(0)\n    err = np.abs(yhat - yte)\n    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))\n    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())\n    print(f\"[prod_bdfe_smiles2d] n={len(d):,} n_feat={len(feats)} test MAE={mae:.3f} \"\n         f\"RMSE={rmse:.3f} R2={r2:.3f}\", flush=True)\n\n    bundle = MODELDIR / f\"bde_surrogate_prod_bdfe_smiles2d_{TAG}.joblib\"\n    joblib.dump({\"members\": [(\"XGB_d8\", xgb8), (\"XGB_d10\", xgb10)], \"imputer\": imp,\n                \"scaler\": sc, \"features\": feats, \"target\": \"bdfe_xtb_kcal (GFN2)\",\n                \"test_mae\": mae, \"test_r2\": r2, \"n_train\": len(tr)}, bundle)\n    print(\"wrote\", bundle, flush=True)\n\n    fig, ax = plt.subplots(figsize=(6, 6))\n    ax.scatter(yte, yhat, s=4, alpha=0.25, color=\"#6a51a3\")\n    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())\n    ax.plot([lo, hi], [lo, hi], \"k--\", lw=1)\n    ax.set_xlabel(\"real xtb product BDFE (kcal/mol)\"); ax.set_ylabel(\"pure-SMILES surrogate prediction (kcal/mol)\")\n    ax.set_title(f\"product BDFE pure-SMILES(2D) surrogate parity (test, n={len(yte):,})\\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\")\n    savefig(f\"130_parity_prod_bdfe_smiles2d_{TAG}.png\")\n\n    imp_gain = xgb8.get_booster().get_score(importance_type=\"gain\")\n    top_feats = sorted(imp_gain.items(), key=lambda kv: kv[1], reverse=True)\n    feat_names = [feats[int(k[1:])] for k, _ in top_feats]\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ax.barh(range(len(top_feats))[::-1], [v for _, v in top_feats])\n    ax.set_yticks(range(len(top_feats))[::-1]); ax.set_yticklabels(feat_names, fontsize=9)\n    ax.set_xlabel(\"XGB_d8 gain importance\")\n    ax.set_title(\"product BDFE pure-SMILES(2D) surrogate feature importance\")\n    savefig(f\"131_feat_importance_prod_bdfe_smiles2d_{TAG}.png\")\n\n    rep = OUT / f\"REPORT_prod_bdfe_smiles2d_surrogate_{TAG}.md\"\n    with open(rep, \"w\") as fh:\n        fh.write(f\"# Product BDFE pure-SMILES (no-xtb) surrogate ({TAG})\\n\\n\")\n        fh.write(f\"Predicts product C-C BDFE (GFN2, {len(d):,} molecules) from ONLY \"\n                f\"{len(feats)} RDKit 2D descriptors computable directly from the SMILES \"\n                f\"string -- no conformer, no xtb at all. Compare to the QM-feature surrogate \"\n                f\"(`train_bde_surrogate.py`, needs xtb geometry+SP): R2=0.901, MAE=2.70.\\n\\n\")\n        fh.write(f\"- **pure-SMILES(2D)**: n_feat={len(feats)} n={len(d):,} MAE={mae:.3f} \"\n                f\"RMSE={rmse:.3f} R2={r2:.3f}\\n\")\n        fh.write(f\"- Top predictive feats: {', '.join(feat_names[:5])}\\n\\n\")\n        gap = 0.901 - r2\n        verdict = (\"substantial (QM electronic-structure info is doing real work; \"\n                  \"not a cheap substitute)\" if gap > 0.15 else\n                  \"small (2D descriptors capture most of the signal; may be usable as a \"\n                  \"cheap prospective pre-filter)\")\n        fh.write(f\"## Interpretation\\n\\nR2 gap vs the QM-feature surrogate: {gap:.3f} -- \"\n                f\"**{verdict}**.\\n\")\n    print(\"wrote\", rep, flush=True)\n    print(\"DONE\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "8bf35a7cfec594b1ce2f9ca57dd59e69e90ed794",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>   "display_title": "train_prod_bdfe_smiles_surrogate.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/analysis/train_bde_surrogate.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"BDE/BDFE surrogate model: predict per-molecule bond dissociation (free) energy directly\nfrom the cheap descriptors already computed for every molecule (72-champion QM feats + the\nmordredslim271 kept mordred subset), instead of always running real xtb (--ohess for BDFE).\nSee bde-surrogate-model-idea memory for the two motivations:\n  1. Redundancy diagnostic -- if a nonlinear model reconstructs BDE/BDFE well from existing\n     feats (high R2), it's mostly redundant info; if it reconstructs poorly, it's genuinely\n     new information the existing feats don't capture (consistent with the earlier\n     correlation analysis showing max |r|~0.5 with any single feature).\n  2. Prospective-scoring speed -- a good surrogate gives near-instant BDE/BDFE for NEW\n     aldehyde/product candidates outside the current 220k library, without running xtb.\n\nTrains 4 independent single-molecule regressors (aldehyde raw-E BDE, aldehyde BDFE, product\nraw-E BDE, product BDFE), each XGB_d8+XGB_d10 ensemble + quantile(05/95) UQ, same 70:20:10\nsplit (seed 42) as the rest of this project. Full diagnostics per target per\ntraining-runs-full-diagnostics memory.\n\"\"\"\nimport json, time\nfrom pathlib import Path\nimport numpy as np, pandas as pd, joblib\nimport matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom xgboost import XGBRegressor\nfrom rdkit import Chem, RDLogger\nfrom rdkit.Chem import rdMolDescriptors, Descriptors\nRDLogger.DisableLog('rdApp.*')\n\nR = \"/scratch-shared/schen3/benzoin-dg\"; H = f\"{R}/data/cross_benzoin/homo_v6\"\nOUT = Path(f\"{H}/viz_gxtb_20260625\"); OUT.mkdir(exist_ok=True)\nTAG = time.strftime(\"%Y%m%d\"); MODELDIR = Path(f\"{R}/pipeline/models\"); MODELDIR.mkdir(exist_ok=True)\n\nPROD_QM = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\n  \"xtb_dipole\",\"mulliken_ketC\",\"mulliken_ketO\",\"mulliken_carbC\",\"mulliken_hydO\",\"mulliken_hydH\",\n  \"wbo_CO_ket\",\"wbo_CC_new\",\"wbo_CO_carb\",\"fukui_plus_ketC\",\"fukui_minus_ketC\",\"dual_ketC\",\n  \"fukui_plus_carbC\",\"fukui_minus_carbC\",\"dual_carbC\",\"vbur_ketC\",\"vbur_carbC\",\"sterimol_L\",\n  \"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\",\"pa_ketO\",\"hb_dist\",\"hb_angle\",\"dih_core\"]\nALD_QM = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\"xtb_dipole\",\n  \"mulliken_CHO_C\",\"mulliken_CHO_O\",\"fukui_plus_CHO_C\",\"fukui_minus_CHO_C\",\"dual_descriptor_CHO_C\",\n  \"wbo_CO\",\"pa_CHO_O\",\"vbur_CHO_C\",\"sterimol_L\",\"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\"]\nGKEYS = [\"TPSA\",\"HBD\",\"HBA\",\"RotB\",\"FracCsp3\",\"nHetero\",\"MolWt\",\"nRing\",\"nAromRing\",\"nAliphRing\",\n         \"nAmide\",\"has_P\",\"has_B\",\"has_S\",\"has_Si\",\"has_halogen\"]\nGLOB = [f\"g_{k}\" for k in GKEYS]\n\n\ndef gfeats(smi):\n    m = Chem.MolFromSmiles(str(smi))\n    if m is None: return {f\"g_{k}\": np.nan for k in GKEYS}\n    s = {a.GetSymbol() for a in m.GetAtoms()}\n    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),\n            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),\n            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),\n            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),\n            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),\n            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]\n    return {f\"g_{k}\": v for k, v in zip(GKEYS, vals)}\n\n\ndef add_global(df, smi_col):\n    u = df[[smi_col]].drop_duplicates()\n    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values\n    return df.merge(g, on=smi_col, how=\"left\")\n\n\ndef _xgb(depth, ne):\n    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,\n                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,\n                        early_stopping_rounds=60, eval_metric=\"mae\")\n\n\ndef make_quantiles():\n    return {q: XGBRegressor(objective=\"reg:quantileerror\", quantile_alpha=q, n_estimators=800,\n                            max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, n_jobs=16)\n            for q in (0.05, 0.95)}\n\n\ndef savefig(name):\n    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches=\"tight\"); plt.close()\n    print(\"wrote\", name, flush=True)\n\n\ndef train_one(df, feats, target, tag, side_label):\n    \"\"\"Train XGB_d8+XGB_d10 ensemble + quantile UQ for a single BDE/BDFE target. Returns dict\n    of results and writes bundle + full diagnostics.\"\"\"\n    d = df.dropna(subset=feats + [target]).reset_index(drop=True)\n    rng = np.random.default_rng(42); idx = rng.permutation(len(d))\n    ntr, nva = int(.7 * len(d)), int(.9 * len(d)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]\n    imp = SimpleImputer(strategy=\"median\").fit(d[feats].values[tr])\n    Xtr_raw, Xva_raw, Xte_raw = imp.transform(d[feats].values[tr]), imp.transform(d[feats].values[va]), imp.transform(d[feats].values[te])\n    sc = StandardScaler().fit(Xtr_raw)\n    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)\n    ytr, yva, yte = d[target].values[tr], d[target].values[va], d[target].values[te]\n\n    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)\n    xgb8.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    xgb10.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    members = [(\"XGB_d8\", xgb8), (\"XGB_d10\", xgb10)]\n    yhat = np.vstack([m.predict(Xte) for _, m in members]).mean(0)\n    err = np.abs(yhat - yte)\n    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))\n    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())\n    print(f\"[{tag}] n={len(d):,} n_feat={len(feats)} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\", flush=True)\n\n    quant = make_quantiles()\n    for q, m in quant.items(): m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)\n    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)\n\n    bundle = MODELDIR / f\"bde_surrogate_{tag}_{TAG}.joblib\"\n    joblib.dump({\"members\": members, \"quantiles\": quant, \"imputer\": imp, \"scaler\": sc,\n                \"features\": feats, \"target\": target, \"test_mae\": mae, \"test_r2\": r2,\n                \"n_train\": len(tr)}, bundle)\n    print(\"wrote\", bundle, flush=True)\n\n    # ── diagnostics ──────────────────────────────────────────────\n    fig, ax = plt.subplots(figsize=(6, 6))\n    ax.scatter(yte, yhat, s=4, alpha=0.25, color=\"#2171b5\")\n    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())\n    ax.plot([lo, hi], [lo, hi], \"k--\", lw=1)\n    ax.set_xlabel(f\"real xtb {target} (kcal/mol)\"); ax.set_ylabel(\"surrogate prediction (kcal/mol)\")\n    ax.set_title(f\"{side_label} surrogate parity (test, n={len(yte):,})\\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\")\n    savefig(f\"110_parity_{tag}_{TAG}.png\")\n\n    resid = yhat - yte\n    fig, ax = plt.subplots(figsize=(7, 5))\n    ax.hist(resid, bins=80, color=\"#6baed6\", edgecolor=\"none\")\n    ax.axvline(0, color=\"k\", lw=1)\n    ax.set_xlabel(\"surrogate - real (kcal/mol)\"); ax.set_ylabel(\"count\")\n    ax.set_title(f\"{side_label} surrogate residual distribution\\nmean={resid.mean():.3f} std={resid.std():.3f}\")\n    savefig(f\"111_residual_hist_{tag}_{TAG}.png\")\n\n    for m, name, idx_fig in [(xgb8, \"XGB_d8\", \"112\"), (xgb10, \"XGB_d10\", \"113\")]:\n        ev = m.evals_result()\n        fig, ax = plt.subplots(figsize=(7, 5))\n        ax.plot(ev[\"validation_0\"][\"mae\"], label=\"train MAE\"); ax.plot(ev[\"validation_1\"][\"mae\"], label=\"val MAE\")\n        ax.axvline(m.best_iteration, color=\"k\", ls=\"--\", lw=1, label=f\"best_iter={m.best_iteration}\")\n        ax.set_xlabel(\"boosting round\"); ax.set_ylabel(\"MAE (kcal/mol)\")\n        ax.set_title(f\"{side_label} {name} training curve\"); ax.legend()\n        savefig(f\"{idx_fig}_{name.lower()}_curve_{tag}_{TAG}.png\")\n\n    imp_gain = xgb8.get_booster().get_score(importance_type=\"gain\")\n    top_feats = sorted(imp_gain.items(), key=lambda kv: kv[1], reverse=True)[:20]\n    feat_names = [feats[int(k[1:])] for k, _ in top_feats]\n    fig, ax = plt.subplots(figsize=(8, 7))\n    ax.barh(range(len(top_feats))[::-1], [v for _, v in top_feats])\n    ax.set_yticks(range(len(top_feats))[::-1]); ax.set_yticklabels(feat_names, fontsize=8)\n    ax.set_xlabel(\"XGB_d8 gain importance\")\n    ax.set_title(f\"{side_label} surrogate top-20 feature importance\")\n    savefig(f\"114_feat_importance_{tag}_{TAG}.png\")\n\n    return {\"tag\": tag, \"target\": target, \"n\": len(d), \"n_feat\": len(feats), \"mae\": mae,\n            \"rmse\": rmse, \"r2\": r2, \"bundle\": str(bundle), \"top_feats\": feat_names[:10]}\n\n\ndef main():\n    kept_mordred = set(json.load(open(f\"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json\"))[\"kept_mordred\"])\n    prod_kept = {c for c in kept_mordred if not c.startswith(\"ald_\")}\n    ald_kept = {c[len(\"ald_\"):] for c in kept_mordred if c.startswith(\"ald_\")}\n\n    # ── aldehyde side ────────────────────────────────────────────\n    a = pd.read_csv(f\"{H}/aldehydes_all.csv\", usecols=[\"id\", \"smiles\"] + ALD_QM, low_memory=False).drop_duplicates(\"id\")\n    a = add_global(a, \"smiles\")\n    ald_header = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", nrows=0).columns\n    ald_want = [\"id\"] + [c for c in ald_header if c in ald_kept]\n    ald_mrd = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", usecols=ald_want, low_memory=False)\n    ald_mrd_cols = [c for c in ald_mrd.columns if c.startswith(\"mordred_\")]\n    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors=\"coerce\")\n    a = a.merge(ald_mrd, on=\"id\", how=\"left\")\n\n    ald_bde = pd.read_csv(f\"{H}/aldehydes_bde_descriptors.csv\", usecols=[\"id\", \"bde_ald_CH_kcal\"])\n    ald_bdfe = pd.read_csv(f\"{H}/aldehydes_bdfe2_descriptors.csv\", usecols=[\"id\", \"bdfe_xtb_kcal\"])\n    ald_bdfe = ald_bdfe[ald_bdfe[\"bdfe_xtb_kcal\"].abs() <= 200]  # drop pathological SCF-failure rows\n\n    a = a.merge(ald_bde, on=\"id\", how=\"left\").merge(ald_bdfe, on=\"id\", how=\"left\")\n    ald_feats = ALD_QM + GLOB + ald_mrd_cols\n    print(f\"aldehyde: n={len(a):,}, feats={len(ald_feats)}\", flush=True)\n    print(f\"  BDE(E) coverage: {a['bde_ald_CH_kcal'].notna().mean()*100:.1f}%  \"\n         f\"BDFE coverage: {a['bdfe_xtb_kcal'].notna().mean()*100:.1f}%\", flush=True)\n\n    # ── product side ─────────────────────────────────────────────\n    p = pd.read_csv(f\"{H}/products_all.csv\", usecols=[\"id\", \"smiles\"] + PROD_QM, low_memory=False).drop_duplicates(\"id\")\n    p = add_global(p, \"smiles\")\n    prod_header = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", nrows=0).columns\n    prod_want = [\"id\"] + [c for c in prod_header if c in prod_kept]\n    prod_mrd = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", usecols=prod_want, low_memory=False)\n    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith(\"mordred_\")]\n    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors=\"coerce\")\n    p = p.merge(prod_mrd, on=\"id\", how=\"left\")\n\n    prod_bde = pd.read_csv(f\"{H}/products_bde_descriptors.csv\", usecols=[\"id\", \"bde_prod_CC_kcal\"])\n    prod_bdfe = pd.read_csv(f\"{H}/products_bdfe2_descriptors.csv\", usecols=[\"id\", \"bdfe_xtb_kcal\"])\n    prod_bdfe = prod_bdfe[prod_bdfe[\"bdfe_xtb_kcal\"].abs() <= 200]\n\n    p = p.merge(prod_bde, on=\"id\", how=\"left\").merge(prod_bdfe, on=\"id\", how=\"left\")\n    prod_feats = PROD_QM + GLOB + prod_mrd_cols\n    print(f\"product: n={len(p):,}, feats={len(prod_feats)}\", flush=True)\n    print(f\"  BDE(E) coverage: {p['bde_prod_CC_kcal'].notna().mean()*100:.1f}%  \"\n         f\"BDFE coverage: {p['bdfe_xtb_kcal'].notna().mean()*100:.1f}%\", flush=True)\n\n    results = {}\n    results[\"ald_bde\"] = train_one(a, ald_feats, \"bde_ald_CH_kcal\", \"ald_bde\", \"aldehyde BDE(raw-E)\")\n    results[\"ald_bdfe\"] = train_one(a, ald_feats, \"bdfe_xtb_kcal\", \"ald_bdfe\", \"aldehyde BDFE\")\n    results[\"prod_bde\"] = train_one(p, prod_feats, \"bde_prod_CC_kcal\", \"prod_bde\", \"product BDE(raw-E)\")\n    results[\"prod_bdfe\"] = train_one(p, prod_feats, \"bdfe_xtb_kcal\", \"prod_bdfe\", \"product BDFE\")\n\n    rep = OUT / f\"REPORT_bde_surrogate_{TAG}.md\"\n    with open(rep, \"w\") as fh:\n        fh.write(f\"# BDE/BDFE surrogate model ({TAG})\\n\\n\")\n        fh.write(\"Predicts per-molecule bond dissociation (free) energy from the existing cheap \"\n                 \"descriptors (72-champion QM + mordredslim271 kept mordred subset), instead of \"\n                 \"running real xtb (--ohess for BDFE). Same 70:20:10 split (seed 42), XGB_d8+\"\n                 \"XGB_d10 ensemble + quantile(05/95) UQ, per target.\\n\\n\")\n        fh.write(\"## Results\\n\\n\")\n        fh.write(\"| target | n | n_feat | test MAE | RMSE | R2 |\\n|---|---|---|---|---|---|\\n\")\n        for tag, r in results.items():\n            fh.write(f\"| {r['target']} ({tag}) | {r['n']:,} | {r['n_feat']} | {r['mae']:.3f} | \"\n                     f\"{r['rmse']:.3f} | {r['r2']:.3f} |\\n\")\n        fh.write(\"\\n## Interpretation\\n\\n\")\n        for tag, r in results.items():\n            redund = \"HIGH (mostly redundant with existing feats)\" if r[\"r2\"] > 0.85 else \\\n                     (\"MODERATE\" if r[\"r2\"] > 0.6 else \"LOW (carries genuinely independent information)\")\n            fh.write(f\"- **{tag}**: R2={r['r2']:.3f} -> redundancy is **{redund}**. \"\n                     f\"Top predictive feats: {', '.join(r['top_feats'][:5])}\\n\")\n        fh.write(\"\\nIf R2 is high enough for practical use (rule of thumb R2>0.85, MAE well under \"\n                 \"the real xtb run's own noise), the corresponding bundle can be used as a fast \"\n                 \"prospective-screening substitute for real xtb BDE/BDFE on new molecules \"\n                 \"outside the current 220k library.\\n\")\n        fh.write(\"\\n## Model bundles\\n\\n\")\n        for tag, r in results.items():\n            fh.write(f\"- `{r['bundle']}`\\n\")\n    json.dump(results, open(OUT / f\"bde_surrogate_results_{TAG}.json\", \"w\"), indent=2)\n    print(\"wrote\", rep, flush=True)\n    print(\"DONE\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "215dc5277fff741d966d4b62014e4aba2163c8c9",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/train_bde_surrogate.py",
>   "display_title": "train_bde_surrogate.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/compute/calc_bde_free_energy_gxtb.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"BDFE, g-xTB-CONSISTENT variant: the GFN2-level BDFE (calc_bde_free_energy_v2.py) gave a\nnull result for the main correction model (see bde-descriptor-idea memory). But the main\nmodel's baseline is g-xTB (dG_gxtb), not GFN2 -- a method mismatch between the BDE/BDFE\ndescriptor (GFN2) and the quantity it's meant to help correct (g-xTB errors) may be diluting\nthe signal. This script computes a method-consistent g-xTB BDFE using this project's own\nestablished hybrid-correction pattern (gxtb_baseline.py): G_gxtb = E_gxtb + (G_gfn2 - E_gfn2),\ni.e. g-xTB electronic energy + GFN2's RRHO thermal correction (g-xTB itself has no native\nproduction thermal-correction pipeline in this project; only geometry+SP were validated,\nsee gxtb-solv-pilot-campaign memory -- though --gxtb --ohess --cosmo works fine per a smoke\ntest, using the SAME hybrid pattern as the rest of this project keeps this new descriptor\nconsistent with dG_gxtb itself, which is what actually matters for it to be diagnostic of\ng-xTB's own errors).\n\nParent's G_gxtb is ALREADY CACHED (G_gxtb column in {aldehydes,products}_all.csv, computed\nby gxtb_baseline.py) -- no new parent calc needed. Only the two radical FRAGMENTS need:\n  1. GFN2 --ohess (as in v2, for the RRHO thermal correction AND to relax the fragment\n     geometry -- xtbopt.xyz)\n  2. g-xTB --sp --cosmo dmso on that SAME relaxed geometry (cheap, no new Hessian)\n  G_gxtb_frag = E_gxtb_frag + (G_gfn2_frag - E_gfn2_frag)\n  BDFE_gxtb = G_gxtb_fragA + G_gxtb_fragB - G_gxtb_parent(cached)\n\nThis redoes the (expensive) fragment ohess step from scratch -- v2's intermediate fragment\nE/G values were never saved, only the final combined bde_E_kcal/bdfe_xtb_kcal (per-molecule\nscratch dirs are deleted immediately after each task, per this project's scratch-hygiene\ndiscipline) -- so there's no way to reuse those results even though this is \"the same\"\nfragment computation. PILOT FIRST before considering a full-library array (same cost as the\noriginal v2 run, ~1 day of core-hours, plus a modest g-xTB SP addendum).\n\nPILOT usage:\n  python calc_bde_free_energy_gxtb.py --which aldehydes --n 20 --out /tmp/bdfe_gxtb_pilot_ald.csv\n\"\"\"\nimport argparse\nimport shutil\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nfrom rdkit import Chem, RDLogger\nfrom rdkit.Chem import rdDetermineBonds\n\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nimport ald_descriptors_qm as A\nimport featurize_product as FP\nimport thermo_orca as Th\nfrom calc_bde_free_energy_v2 import g_h_atom_correction, H_ATOM_G_CORR, mol_with_bonds, split_at_bond, _xyz_block\n\nRDLogger.DisableLog(\"rdApp.*\")\nHARTREE_TO_KCAL = 627.509474\n\n\ndef _run_ohess_uhf(xyz_str, work_dir, xtb_bin, charge, uhf, solvent=\"dmso\", T=298.15, P_atm=1.0, timeout=1800):\n    work_dir.mkdir(parents=True, exist_ok=True)\n    (work_dir / \"mol.xyz\").write_text(xyz_str, encoding=\"utf-8\")\n    Th._write_xtb_inp(work_dir / \"xtb.inp\", T, P_atm * 1.01325)\n    cmd = [xtb_bin, \"mol.xyz\", \"--gfn\", \"2\", \"--ohess\", \"tight\", \"--input\", \"xtb.inp\",\n          \"--chrg\", str(charge), \"--uhf\", str(uhf), \"--norestart\", \"--parallel\", \"1\"]\n    if solvent:\n        cmd += [\"--alpb\", solvent]\n    stdout = \"\"\n    try:\n        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)\n        stdout = r.stdout + r.stderr\n    except subprocess.TimeoutExpired:\n        pass\n    return stdout\n\n\ndef _gxtb_sp(xyz_file, work_dir, gxtb_bin, charge, uhf, solvent=\"dmso\", timeout=300):\n    work_dir.mkdir(parents=True, exist_ok=True)\n    shutil.copy(xyz_file, work_dir / \"g.xyz\")\n    cmd = [gxtb_bin, \"g.xyz\", \"--gxtb\", \"--sp\", \"--chrg\", str(charge), \"--uhf\", str(uhf)]\n    if solvent:\n        cmd += [\"--cosmo\", solvent]\n    try:\n        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)\n        return A._parse_xtb_energy(r.stdout + r.stderr)\n    except subprocess.TimeoutExpired:\n        return None\n\n\ndef _frag_G_E_gxtb(symbols, coords, work_dir, xtb_bin, gxtb_bin, solvent=\"dmso\") -> dict:\n    \"\"\"Fresh dual-level calc for a RADICAL fragment: GFN2 --ohess (thermal correction +\n    relaxed geometry) followed by a g-xTB SP on that SAME relaxed geometry.\"\"\"\n    if len(symbols) == 1:  # isolated H atom -- no relaxation needed, no ohess possible\n        wd = work_dir / \"h_sp\"; wd.mkdir(parents=True, exist_ok=True)\n        (wd / \"h.xyz\").write_text(_xyz_block(symbols, coords))\n        e_gxtb = _gxtb_sp(wd / \"h.xyz\", wd / \"gxtb\", gxtb_bin, charge=0, uhf=1, solvent=solvent)\n        if e_gxtb is None:\n            return {\"G_gxtb\": None, \"E_gxtb\": None}\n        return {\"G_gxtb\": e_gxtb + H_ATOM_G_CORR, \"E_gxtb\": e_gxtb}  # method-independent ideal-gas correction\n\n    ohess_dir = work_dir / \"ohess\"\n    stdout = _run_ohess_uhf(_xyz_block(symbols, coords), ohess_dir, xtb_bin, charge=0, uhf=1, solvent=solvent)\n    e_gfn2 = Th._parse_xtb_energy(stdout)\n    g_gfn2 = Th.parse_xtb_G(stdout)\n    opt_xyz = ohess_dir / \"xtbopt.xyz\"\n    if e_gfn2 is None or g_gfn2 is None or not opt_xyz.exists():\n        return {\"G_gxtb\": None, \"E_gxtb\": None}\n    e_gxtb = _gxtb_sp(opt_xyz, work_dir / \"gxtb\", gxtb_bin, charge=0, uhf=1, solvent=solvent)\n    if e_gxtb is None:\n        return {\"G_gxtb\": None, \"E_gxtb\": None}\n    g_gxtb = e_gxtb + (g_gfn2 - e_gfn2)  # project's established hybrid-correction pattern\n    return {\"G_gxtb\": g_gxtb, \"E_gxtb\": e_gxtb}\n\n\ndef bdfe_gxtb_aldehyde_ch(xyz_file, parent_G_gxtb, parent_E_gxtb, xtb_bin, gxtb_bin, work_dir) -> dict:\n    row = {\"bdfe_gxtb_kcal\": np.nan, \"bde_gxtb_kcal\": np.nan}\n    if parent_G_gxtb is None:\n        return row\n    xyz = Path(xyz_file).read_text()\n    symbols, coords = A.parse_xyz(xyz)\n    hits = A.find_aldehyde_atoms(symbols, coords)\n    if not hits:\n        return row\n    c_idx, o_idx, _ = hits[0]\n    dist = np.linalg.norm(coords - coords[c_idx], axis=1)\n    h_idx = next((k for k in range(len(symbols))\n                 if symbols[k] == \"H\" and dist[k] < A.CH_BOND_MAX and k != c_idx), None)\n    if h_idx is None:\n        return row\n    mol = mol_with_bonds(xyz_file)\n    if mol is None:\n        return row\n    split = split_at_bond(mol, c_idx, h_idx, coords, symbols)\n    if split is None:\n        return row\n    (symA, coordA), (symB, coordB) = split\n    if len(symB) != 1:\n        (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)\n    if len(symB) != 1:\n        return row\n\n    fragA = _frag_G_E_gxtb(symA, coordA, work_dir / \"fragA\", xtb_bin, gxtb_bin)\n    fragB = _frag_G_E_gxtb(symB, coordB, work_dir / \"fragB\", xtb_bin, gxtb_bin)\n    if None not in (fragA[\"G_gxtb\"], fragB[\"G_gxtb\"]):\n        row[\"bdfe_gxtb_kcal\"] = round((fragA[\"G_gxtb\"] + fragB[\"G_gxtb\"] - parent_G_gxtb) * HARTREE_TO_KCAL, 3)\n    if parent_E_gxtb is not None and None not in (fragA[\"E_gxtb\"], fragB[\"E_gxtb\"]):\n        row[\"bde_gxtb_kcal\"] = round((fragA[\"E_gxtb\"] + fragB[\"E_gxtb\"] - parent_E_gxtb) * HARTREE_TO_KCAL, 3)\n    return row\n\n\ndef bdfe_gxtb_product_cc(xyz_file, parent_G_gxtb, parent_E_gxtb, xtb_bin, gxtb_bin, work_dir) -> dict:\n    row = {\"bdfe_gxtb_kcal\": np.nan, \"bde_gxtb_kcal\": np.nan}\n    if parent_G_gxtb is None:\n        return row\n    xyz = Path(xyz_file).read_text()\n    symbols, coords = A.parse_xyz(xyz)\n    core = FP.find_benzoin_core(symbols, coords)\n    if core is None:\n        return row\n    i, j = core[\"ketC\"], core[\"carbC\"]\n    mol = mol_with_bonds(xyz_file)\n    if mol is None:\n        return row\n    split = split_at_bond(mol, i, j, coords, symbols)\n    if split is None:\n        return row\n    (symA, coordA), (symB, coordB) = split\n\n    fragA = _frag_G_E_gxtb(symA, coordA, work_dir / \"fragA\", xtb_bin, gxtb_bin)\n    fragB = _frag_G_E_gxtb(symB, coordB, work_dir / \"fragB\", xtb_bin, gxtb_bin)\n    if None not in (fragA[\"G_gxtb\"], fragB[\"G_gxtb\"]):\n        row[\"bdfe_gxtb_kcal\"] = round((fragA[\"G_gxtb\"] + fragB[\"G_gxtb\"] - parent_G_gxtb) * HARTREE_TO_KCAL, 3)\n    if parent_E_gxtb is not None and None not in (fragA[\"E_gxtb\"], fragB[\"E_gxtb\"]):\n        row[\"bde_gxtb_kcal\"] = round((fragA[\"E_gxtb\"] + fragB[\"E_gxtb\"] - parent_E_gxtb) * HARTREE_TO_KCAL, 3)\n    return row\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\"--which\", choices=[\"aldehydes\", \"products\"], required=True)\n    ap.add_argument(\"--n\", type=int, default=None)\n    ap.add_argument(\"--chunk-id\", type=int, default=None)\n    ap.add_argument(\"--chunk-size\", type=int, default=150)\n    ap.add_argument(\"--out\", default=None)\n    ap.add_argument(\"--out-dir\", default=None)\n    ap.add_argument(\"--xtb-bin\", default=\"/home/schen3/xtb/bin/xtb\")\n    ap.add_argument(\"--gxtb-bin\", default=\"/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb\")\n    ap.add_argument(\"--work-dir\", default=None)\n    args = ap.parse_args()\n\n    H = Path(\"/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6\")\n    src = H / f\"{args.which}_all.csv\"\n    # G_xtb/xtb_energy (GFN2 G/E, cached from the original funnel_v3 featurization) let us\n    # algebraically recover the parent's raw g-xTB electronic energy for FREE, no new xtb\n    # call needed: E_gxtb = G_gxtb - (G_gfn2 - E_gfn2), inverting gxtb_baseline.py's own\n    # G_gxtb = E_gxtb + (G_gfn2 - E_gfn2) hybrid-correction formula. This gives raw-E\n    # bde_gxtb_kcal alongside bdfe_gxtb_kcal at essentially no extra compute cost.\n    df = pd.read_csv(src, usecols=[\"id\", \"xyz_file\", \"error\", \"G_gxtb\", \"G_xtb\", \"xtb_energy\"], dtype=str,\n                     keep_default_na=False, low_memory=False)\n    df = df[(df[\"error\"] == \"\") & (df[\"xyz_file\"] != \"\") & (df[\"G_gxtb\"] != \"\")\n           & (df[\"G_xtb\"] != \"\") & (df[\"xtb_energy\"] != \"\")]\n    df = df.drop_duplicates(\"id\").reset_index(drop=True)\n    df[\"G_gxtb\"] = df[\"G_gxtb\"].astype(float)\n    df[\"G_xtb\"] = df[\"G_xtb\"].astype(float)\n    df[\"xtb_energy\"] = df[\"xtb_energy\"].astype(float)\n    df[\"E_gxtb_parent\"] = df[\"G_gxtb\"] - (df[\"G_xtb\"] - df[\"xtb_energy\"])\n\n    if args.chunk_id is not None:\n        lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))\n        if lo >= len(df):\n            print(f\"chunk {args.chunk_id}: out of range, nothing to do\", flush=True); return\n        df = df.iloc[lo:hi].reset_index(drop=True)\n        out_path = Path(args.out_dir) / f\"chunk_{args.chunk_id:04d}.csv\"\n        out_path.parent.mkdir(parents=True, exist_ok=True)\n        if out_path.exists():\n            existing = pd.read_csv(out_path, usecols=[\"id\"])\n            if len(existing) >= len(df):\n                print(f\"chunk {args.chunk_id}: already done ({len(existing)}/{len(df)}) -- skip\", flush=True)\n                return\n        print(f\"BDFE-gxtb {args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(df)})\", flush=True)\n    else:\n        df = df.head(args.n or 15)\n        out_path = Path(args.out)\n        print(f\"BDFE-gxtb pilot on {len(df)} {args.which}\", flush=True)\n\n    wd_root = Path(args.work_dir or \"/tmp/bdfe_gxtb_pilot\")\n    rows = []\n    for _, rec in df.iterrows():\n        wd = wd_root / f\"m{rec['id']}\"\n        try:\n            if args.which == \"aldehydes\":\n                r = bdfe_gxtb_aldehyde_ch(rec[\"xyz_file\"], rec[\"G_gxtb\"], rec[\"E_gxtb_parent\"], args.xtb_bin, args.gxtb_bin, wd)\n            else:\n                r = bdfe_gxtb_product_cc(rec[\"xyz_file\"], rec[\"G_gxtb\"], rec[\"E_gxtb_parent\"], args.xtb_bin, args.gxtb_bin, wd)\n        except Exception as e:\n            r = {\"error\": str(e)}\n        r[\"id\"] = rec[\"id\"]\n        rows.append(r)\n        print(rec[\"id\"], r, flush=True)\n        shutil.rmtree(wd, ignore_errors=True)\n\n    result = pd.DataFrame(rows)\n    result.to_csv(out_path, index=False)\n    ok_g = result[\"bdfe_gxtb_kcal\"].notna().sum() if \"bdfe_gxtb_kcal\" in result else 0\n    ok_e = result[\"bde_gxtb_kcal\"].notna().sum() if \"bde_gxtb_kcal\" in result else 0\n    print(f\"wrote {out_path}  bdfe_gxtb:{ok_g}/{len(result)}  bde_gxtb:{ok_e}/{len(result)}\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "afd45b87836dd28c6f12e05b555ddfa88b0e6659",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/compute/calc_bde_free_energy_gxtb.py",
>   "display_title": "calc_bde_free_energy_gxtb.py"
> }
> ```
>
> </details>
>
> 仓库信息显示，这里的 BDE/BDFE 并不是一个通用“键能预测器”，而是苯偶姻反应 ΔG 模型中的四个物理描述符（醛 C–H 与产物新生 C–C，各含 BDE/BDFE）。这会是与 ALFABET（你写作 ALPHABET）比较时最重要的口径差异；我正在核对 2026 论文是否解决了相同层次的问题。
>
> <details><summary>Called 9 tools, searched the web</summary>
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `https://doi.org/10.1038/s41467-026-74919-8`
> - Searched the web for `"s41467-026-74919-8" ...`
> - Searched the web for `site:pubs.acs.org "Transferable machine learning interatomic potential for bond dissociation energy prediction" ...`
>
> </details>
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "train_test_split",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 20
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/train_cb_721.py",
>       "path": "pipeline/train_cb_721.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/train_cb_721.py",
>       "display_title": "pipeline/train_cb_721.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/DATA_SPLITS_screen_v6_vs_homo_v6.md",
>       "path": "pipeline/docs/DATA_SPLITS_screen_v6_vs_homo_v6.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/DATA_SPLITS_screen_v6_vs_homo_v6.md",
>       "display_title": "pipeline/docs/DATA_SPLITS_screen_v6_vs_homo_v6.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/learning_curve_cb.py",
>       "path": "pipeline/learning_curve_cb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/learning_curve_cb.py",
>       "display_title": "pipeline/learning_curve_cb.py"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "MORDREDSLIM271_BDEGXTB",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 20
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "path": "pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "display_title": "pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_finalize_mordredslim271_bdegxtb.sh",
>       "path": "pipeline/slurm/submit_finalize_mordredslim271_bdegxtb.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_finalize_mordredslim271_bdegxtb.sh",
>       "display_title": "pipeline/slurm/submit_finalize_mordredslim271_bdegxtb.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707_zh.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707_zh.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_shap_mordredslim271_bdegxtb_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "path": "pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "display_title": "pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "path": "README.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "display_title": "README.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/plot_noise_band_champion275.py",
>       "path": "pipeline/analysis/plot_noise_band_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/plot_noise_band_champion275.py",
>       "display_title": "pipeline/analysis/plot_noise_band_champion275.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_shap_slim_mordredslim271_bdegxtb.sh",
>       "path": "pipeline/slurm/submit_shap_slim_mordredslim271_bdegxtb.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_shap_slim_mordredslim271_bdegxtb.sh",
>       "display_title": "pipeline/slurm/submit_shap_slim_mordredslim271_bdegxtb.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_final_model_decision_20260707_zh.md",
>       "path": "pipeline/docs/REPORT_final_model_decision_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_final_model_decision_20260707_zh.md",
>       "display_title": "pipeline/docs/REPORT_final_model_decision_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_final_model_decision_20260707_en.md",
>       "path": "pipeline/docs/REPORT_final_model_decision_20260707_en.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_final_model_decision_20260707_en.md",
>       "display_title": "pipeline/docs/REPORT_final_model_decision_20260707_en.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_smiles_dft_full.py",
>       "path": "pipeline/analysis/gnn_smiles_dft_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_smiles_dft_full.py",
>       "display_title": "pipeline/analysis/gnn_smiles_dft_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_model_attempts_weekly_20260706_en.md",
>       "path": "pipeline/docs/REPORT_model_attempts_weekly_20260706_en.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_model_attempts_weekly_20260706_en.md",
>       "display_title": "pipeline/docs/REPORT_model_attempts_weekly_20260706_en.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_model_attempts_weekly_20260706_zh.md",
>       "path": "pipeline/docs/REPORT_model_attempts_weekly_20260706_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_model_attempts_weekly_20260706_zh.md",
>       "display_title": "pipeline/docs/REPORT_model_attempts_weekly_20260706_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/aromatic_subset_analysis_champion275.py",
>       "path": "pipeline/analysis/aromatic_subset_analysis_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/aromatic_subset_analysis_champion275.py",
>       "display_title": "pipeline/analysis/aromatic_subset_analysis_champion275.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/deep_error_analysis_champion275.py",
>       "path": "pipeline/analysis/deep_error_analysis_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/deep_error_analysis_champion275.py",
>       "display_title": "pipeline/analysis/deep_error_analysis_champion275.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "path": "pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/chemberta_dft_full.py",
>       "path": "pipeline/analysis/chemberta_dft_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/chemberta_dft_full.py",
>       "display_title": "pipeline/analysis/chemberta_dft_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "path": "pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/METHODS_BDE_gxtb_20260707_en.md",
>       "display_title": "pipeline/docs/METHODS_BDE_gxtb_20260707_en.md"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# MORDREDSLIM271_BDEGXTB production model (20260706)\n\n- Features: **275** (mordredslim271's 271 [72 champion QM + 199 targeted mordred] + 4 g-xTB-consistent BDE/BDFE: prod_bdfe_gxtb_kcal, ald_bdfe_gxtb_kcal, prod_bde_gxtb_kcal, ald_bde_gxtb_kcal)\n- **TEST MAE 1.503, RMSE 2.257, R2 0.875** vs mordredslim271 production champion (test MAE 1.525) and the quick 2-member XGB check that motivated this run (1.612 -> 1.563, REPORT_bdfe_gxtb_full_augment_20260706.md)\n- Uncertainty routing: confident 85% MAE 1.252 | routed MAE 2.923 | ROC AUC 0.796\n- Model: `/scratch-shared/schen3/benzoin-dg/pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`\n- Full-library output: `products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv` (218,227 mols; 33,275 flagged route_to_dft)\n",
>   "encoding": "utf-8",
>   "sha": "47b1103e769577cebf57c1d4ed2933acf75d64a8",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md",
>   "display_title": "REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/docs/REPORT_final_model_decision_20260707_zh.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# 最终模型决策：MORDREDSLIM271_BDEGXTB —— 2026-07-07 后续分析批次综合总结\n\n## 结论\n\n**MORDREDSLIM271_BDEGXTB 仍是生产冠军模型**（275 特征：72 个 QM + 199 个 SHAP 精简 mordred +\n4 个 g-xTB BDE/BDFE；MLP + XGB_d8 + XGB_d10 + 分位数不确定性集成；测试集 MAE **1.503**，\nRMSE 2.257，R² 0.875；bundle 路径\n`pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`）。为了\n(a) 对该模型做健全性检查与解读，(b) 尝试在架构层面（包括用 GNN）超越它，共进行了四项后续\n分析。没有一项真正超越它；以下各节总结每项分析的发现，以及为何该决策维持不变。\n\n## 1. SHAP 成本审计——保留模型，简化未来的特征获取\n\n在 4000 行测试子样本上做 SHAP 分析（XGB_d8）显示，4 个 g-xTB BDE/BDFE 特征的用处并不对等：\n`ald_bde_gxtb_kcal`（排名 4/275）和 `prod_bde_gxtb_kcal`（排名 6/275）——两者都便宜（单次\nSP/opt）——的 SHAP 重要性求和是 `prod_bdfe_gxtb_kcal`（排名15）和 `ald_bdfe_gxtb_kcal`\n（排名38）的 **3.7 倍**，而后两者需要对每个片段做昂贵的完整 Hessian `--ohess`。**行动建议：\n未来对新分子做前瞻性筛选时，只计算 BDE，跳过 BDFE**——现有 bundle 两者都保留（全 22 万库已经\n算过，没有理由为了删掉 2 个已算好的列而重新训练），但这一发现排除了未来在 BDFE 族（热/熵）\n描述符上继续投入的必要性。完整方法论见\n`REPORT_shap_mordredslim271_bdegxtb_20260707.md` / `_zh.md` 以及\n[METHODS_BDE_gxtb_20260707_zh.md](METHODS_BDE_gxtb_20260707_zh.md)。\n\n## 2. 深度误差/噪声带分析——剩余 1.503 kcal/mol 误差从何而来\n\n32.3% 的测试误差（21,910 个分子中的 7,075 个）真实地高于 baseline_72 重新打乱种子研究建立的\n噪声地板（1.571±0.013 kcal/mol，3-sigma 截断值 1.610）——其余约 68% 的误差落在纯 DFT 标签\n噪声可以解释的范围内，无法靠更好的模型修复。在真实误差长尾中：**磺酰基富集 11.2 倍，含磷\n9.5 倍，亚胺 3.6 倍，酰胺 3.4 倍（新发现的驱动因素），酯 1.8 倍，硝基 1.5 倍**。结构空间\n（Morgan 指纹）聚类发现 4 个相似的、以卤素为主导的困难簇（平均误差 2.8-3.0）+ 1 个以酰胺为\n主导的簇（3.36，最差）+ 1 个无主导标签的异质簇——与此前 SHAP 归因空间\"4 种异质失败模式\"的\n发现一致。**含义：如果要继续追求精度提升，应针对这些具体的官能团失败模式（例如专门的磺酰基/\n酰胺特征，或有针对性的 DFT 重新标注），而不是泛泛的模型/架构改动**——见新增的噪声带直方图\n`126_noise_band_histogram_champion275_20260707.png`。\n\n## 3. 芳香族子集深入分析——历史采样偏差已解决，脂肪族现在是更难的部分\n\n全库芳香族占比（66.3%）与测试集占比（66.3%，差距 0.03 个百分点）一致——旧有的 MaxMin\n主动选择对芳香族的欠采样偏差在当前全库训练群体中**已不存在**（该问题特定于早期的主动选择\n阶段，如今 DFT-SP 已覆盖近乎全部 clean_v4 库，该问题已不成立）。按范围划分的精度：**芳香族\nMAE 1.327/R²0.875，脂肪族 MAE 1.870/R²0.842**——脂肪族现在是更难的部分，不确定度路由也正确\n反映了这一点（route-to-DFT 比例 26.5% 对 9.7%）。芳香族内部，化学型的影响很小（简单碳环\n1.268 最好，嗪/唑类含氮 1.391，呋喃 1.375，噻吩 1.275——都很接近）。**含义：如果优先安排\n进一步的数据采集，脂肪族分子是价值更高的目标，而非芳香族。**\n\n## 4. GNN 追平尝试——确认架构不是关键杠杆\n\n`gnn_dual_qm_champion275.py`（任务 24478591）给双编码器 GINE 架构（完整架构扫描中表现最好\n的 GNN 架构，`REPORT_homo_v6_gnn_architectures_20260629.md`）在读出层喂入与冠军模型完全\n相同的 275 个特征，在同一个匹配约 21.8 万分子群体上使用相同的 70:20:10 种子 42 划分协议——\n测试的是更丰富的特征（而非不同的架构）能否让 GNN 追平或超越 1.503。**结果：测试集 MAE\n1.552**（RMSE 2.269，R²0.874，54.4 万参数）——大幅缩小了此前的差距（dual_qm+56特征版本是\n1.646，如今用 275 特征降到 1.552），但仍落后表格模型冠军 0.049。结合此前完整架构扫描\n（不论用哪种卷积算子，所有 2D 卷积算子都在 2.58-2.68 见顶；只有注入 QM 特征才能打破这个\n平台期）以及更早的 3D-GNN 结果（SchNet 2.18，输给同样数据下的 MLP 1.83），这一结论已在三个\n独立测试中得到稳健验证：**即使给 GNN 与表格模型完全相同的信息，它依然无法像梯度提升树那样\n充分利用这些信息。** 对这个任务而言，架构替换不是关键杠杆，信息（特征）才是。\n\n## 5. 架构提升尝试：GNN + 表格模型堆叠集成\n\n既然不同架构和特征对齐都未能让 GNN 单独获胜，架构层面剩下的杠杆是**组合**这两类模型家族，\n而非用一个取代另一个——GNN（消息传递）和 GBDT（轴对齐分裂）是足够不同的函数类别，它们的\n误差可能只是部分相关，若如此，即使两者都不能单独获胜，组合也可能超越两者。\n`gnn_dual_qm_champion275_ensemble.py`（任务 24482531）用完全相同的协议重新训练了同一个\n双编码器 GNN，同时额外保存了带分子 ID 的逐分子测试集预测，再与表格模型冠军已保存的测试集预测\n（`test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv`）在权重扫描\n`w_gnn ∈ {0, 0.05, ..., 1.0}` 上做混合。\n\n**中途发现并修复了一个方法论 bug（任务 24482531）：第一次尝试的混合对比结果是无效的。**\n两个脚本都用 `np.random.default_rng(42).permutation(N)` 做 70:20:10 划分，但各自独立推导\n自己的群体规模 `N`（GNN 脚本与表格模型最终脚本的缺失值处理略有不同）——即使种子相同，对不同\nN 做排列本质上是两个不相关的随机序列，所以各自选出的\"最后 10%\"测试集几乎没有交集（任务\n24482531 的运行中断言失败：`unexpectedly small overlap`，退出码 1，正是证实了这一点）。\n**修复方法**（`gnn_ensemble_reeval_matched.py`）：不再依赖 GNN 自身独立划分恰好对齐，而是\n**复用已训练好的 GNN 权重（24482531 保存），直接在表格模型测试集的确切分子 ID 上重新评估**。\n这一修复还暴露出一个更严重、必须过滤掉的风险：在表格模型 21,858 个测试集 ID 中，**有 15,258\n个（70%）其实落在了 GNN 自己的训练集里**——这正是两个独立划分对同一群体取交集时按概率预期\n应有的约 70% 重合率——必须先剔除这些，否则 GNN 对它们的\"记忆\"（而非泛化）预测会虚高其表面\n精度。\n\n**在干净、无泄漏的交集上的结果（n=6,613 个分子，任务 24489591）：**\n\n| 配置 | MAE（n=6,613 子集） |\n|---|---|\n| 仅表格模型 | 1.476 |\n| 仅 GNN | 1.557 |\n| **最佳混合（w_gnn=0.40，即 60% 表格模型 + 40% GNN）** | **1.425** |\n\n相对仅表格模型在该子集上的差值：**-0.051**，明显超出已确立的噪声带（0.02-0.03 kcal/mol）\n——混合确实带来了真实提升，与\"GNN 和 GBDT 的误差只是部分相关\"的假设一致。**需要说明的\n局限**：这个结果是在约 6,600 个分子的子集（无泄漏交集）上测得的，而非完整的 21,910 行官方\n测试集，且该子集上仅表格模型的 MAE（1.476）与完整测试集上官方报告的 1.503 并不相同——所以\n这个 -0.051 的差值只在该子集内部比较时严格成立，尚不能直接断言完整生产模型会从 1.503 降到\n约 1.45。**采纳前需要先确认的下一步**：从头开始，用一个共享、对齐的 70:20:10 划分（两个\n模型采用同一套基于 ID 的折分配）重新训练两个模型，这样就能在完整测试集上做混合，而不需要\n这种子集化处理，并确认该提升在全量规模下依然成立。\n\n## 总体建议\n\n目前仍将 MORDREDSLIM271_BDEGXTB 作为唯一的生产模型，但**堆叠集成的结果足够有希望（-0.051\nMAE，约为噪声带的 2 倍，且是在真实但部分重合的交集上得到的），值得先做上述对齐划分的重新\n训练**，再决定是否在生产打分中采用混合预测器（代价是推理时也要额外运行 GNN）。如果希望在\n集成结果之外进一步提升精度，还有两条有证据支持的路径：(a) 针对第 2 节中官能团失败模式\n（磺酰基/酰胺/亚胺/磷）做定向工作，(b) 在未来任何标注工作中优先考虑脂肪族分子（第 3 节）。\n不建议继续投入独立 GNN 替代方案（第 4 节）或 BDFE 族描述符（第 1 节）。\n",
>   "encoding": "utf-8",
>   "sha": "2b51cdc87e1749ef76da40f57bb6af299d74085e",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/docs/REPORT_final_model_decision_20260707_zh.md",
>   "display_title": "REPORT_final_model_decision_20260707_zh.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"FULL PRODUCTION run of the new champion candidate: MORDREDSLIM271 (72 champion QM +\n199 SHAP-pruned mordred, test MAE 1.525) + the 4 g-xTB-consistent BDE/BDFE features\n(ald_bde_gxtb_kcal, ald_bdfe_gxtb_kcal, prod_bde_gxtb_kcal, prod_bdfe_gxtb_kcal) that beat\nthe noise band on the quick 2-member XGB check (1.612 -> 1.563, see\nfinalize_correction_bdfe_gxtb_full.py / REPORT_bdfe_gxtb_full_augment_20260706.md).\n\nThis mirrors finalize_correction_mordred_slim.py's exact production treatment (MLP + 2xXGB\nensemble + quantile-UQ uncertainty routing + full diagnostics) instead of that quick-check's\nbare 2-member XGB comparison, to get the real production-grade MAE number on 275 features.\n\"\"\"\nimport json, time\nfrom pathlib import Path\nimport numpy as np, pandas as pd, joblib\nimport matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt\nfrom sklearn.neural_network import MLPRegressor\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import roc_curve, auc\nfrom xgboost import XGBRegressor\nfrom rdkit import Chem, RDLogger\nfrom rdkit.Chem import Draw, rdMolDescriptors, Descriptors\nRDLogger.DisableLog('rdApp.*')\n\nR = \"/scratch-shared/schen3/benzoin-dg\"; H = f\"{R}/data/cross_benzoin/homo_v6\"\nOUT = Path(f\"{H}/viz_gxtb_20260625\"); OUT.mkdir(exist_ok=True)\nTAG = time.strftime(\"%Y%m%d\"); MODELDIR = Path(f\"{R}/pipeline/models\"); MODELDIR.mkdir(exist_ok=True)\nROUTE_FRAC = 0.15\nNAME = \"MORDREDSLIM271_BDEGXTB\"\n\nPROD_QM = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\n  \"xtb_dipole\",\"mulliken_ketC\",\"mulliken_ketO\",\"mulliken_carbC\",\"mulliken_hydO\",\"mulliken_hydH\",\n  \"wbo_CO_ket\",\"wbo_CC_new\",\"wbo_CO_carb\",\"fukui_plus_ketC\",\"fukui_minus_ketC\",\"dual_ketC\",\n  \"fukui_plus_carbC\",\"fukui_minus_carbC\",\"dual_carbC\",\"vbur_ketC\",\"vbur_carbC\",\"sterimol_L\",\n  \"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\",\"pa_ketO\",\"hb_dist\",\"hb_angle\",\"dih_core\"]\nALD = [\"xtb_HOMO\",\"xtb_LUMO\",\"xtb_gap\",\"xtb_IP\",\"xtb_EA\",\"xtb_mu\",\"xtb_eta\",\"xtb_omega\",\"xtb_dipole\",\n  \"mulliken_CHO_C\",\"mulliken_CHO_O\",\"fukui_plus_CHO_C\",\"fukui_minus_CHO_C\",\"dual_descriptor_CHO_C\",\n  \"wbo_CO\",\"pa_CHO_O\",\"vbur_CHO_C\",\"sterimol_L\",\"sterimol_B1\",\"sterimol_B5\",\"SASA_total\",\"P_int\"]\nALDp = [f\"ald_{c}\" for c in ALD]\nGKEYS = [\"TPSA\",\"HBD\",\"HBA\",\"RotB\",\"FracCsp3\",\"nHetero\",\"MolWt\",\"nRing\",\"nAromRing\",\"nAliphRing\",\n         \"nAmide\",\"has_P\",\"has_B\",\"has_S\",\"has_Si\",\"has_halogen\"]\nGLOB = [f\"g_{k}\" for k in GKEYS]\nFEATS_72 = PROD_QM + ALDp + GLOB\nBDE_COLS = [\"prod_bdfe_gxtb_kcal\", \"ald_bdfe_gxtb_kcal\", \"prod_bde_gxtb_kcal\", \"ald_bde_gxtb_kcal\"]\n\n\ndef gfeats(smi):\n    m = Chem.MolFromSmiles(str(smi))\n    if m is None: return {f\"g_{k}\": np.nan for k in GKEYS}\n    s = {a.GetSymbol() for a in m.GetAtoms()}\n    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),\n            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),\n            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),\n            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),\n            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),\n            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]\n    return {f\"g_{k}\": v for k, v in zip(GKEYS, vals)}\n\n\ndef add_global(df, smi_col):\n    u = df[[smi_col]].drop_duplicates()\n    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values\n    return df.merge(g, on=smi_col, how=\"left\")\n\n\ndef _xgb(depth, ne):\n    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,\n                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,\n                        early_stopping_rounds=60, eval_metric=\"mae\")\n\n\ndef make_quantiles():\n    return {q: XGBRegressor(objective=\"reg:quantileerror\", quantile_alpha=q, n_estimators=800,\n                            max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, n_jobs=16)\n            for q in (0.05, 0.95)}\n\n\ndef savefig(name):\n    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches=\"tight\"); plt.close()\n    print(\"wrote\", name, flush=True)\n\n\ndef main():\n    cons = Path(f\"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet\")\n    dft = pd.read_parquet(cons, columns=[\"id\", \"dG_orca_kcal\"]).dropna(subset=[\"dG_orca_kcal\"]).drop_duplicates(\"id\", keep=\"last\")\n\n    p = pd.read_csv(f\"{H}/products_all.csv\", usecols=[\"id\", \"donor_id\", \"smiles\", \"dG_gxtb_kcal\"] + PROD_QM, low_memory=False)\n    a = pd.read_csv(f\"{H}/aldehydes_all.csv\", usecols=[\"id\", \"smiles\"] + ALD, low_memory=False).drop_duplicates(\"id\")\n    a_r = a.rename(columns={\"id\": \"ald_id\", \"smiles\": \"ald_smiles\", **{c: f\"ald_{c}\" for c in ALD}})\n    cls = pd.read_parquet(f\"{H}/aldehyde_class.parquet\")\n\n    # SHAP-pruned mordredslim271 selection (job 24405069, REPORT_mordred_slim_20260703.md)\n    kept_mordred = set(json.load(open(f\"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json\"))[\"kept_mordred\"])\n    prod_kept = [c for c in kept_mordred if not c.startswith(\"ald_\")]\n    ald_kept_raw = [c[len(\"ald_\"):] for c in kept_mordred if c.startswith(\"ald_\")]\n\n    prod_header = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", nrows=0).columns\n    prod_want = [\"id\"] + [c for c in prod_header if c in prod_kept]\n    prod_mrd = pd.read_csv(f\"{H}/products_mordred_descriptors.csv\", usecols=prod_want, low_memory=False)\n    ald_header = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", nrows=0).columns\n    ald_want = [\"id\"] + [c for c in ald_header if c in ald_kept_raw]\n    ald_mrd = pd.read_csv(f\"{H}/aldehydes_mordred_descriptors.csv\", usecols=ald_want, low_memory=False)\n\n    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith(\"mordred_\")]\n    ald_mrd = ald_mrd.rename(columns={\"id\": \"ald_id\"})\n    ald_mrd_cols_raw = [c for c in ald_mrd.columns if c.startswith(\"mordred_\")]\n    ald_mrd = ald_mrd.rename(columns={c: f\"ald_{c}\" for c in ald_mrd_cols_raw})\n    ald_mrd_cols = [f\"ald_{c}\" for c in ald_mrd_cols_raw]\n    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors=\"coerce\")\n    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors=\"coerce\")\n    print(f\"slim mordred: {len(prod_mrd_cols)+len(ald_mrd_cols)}/199 kept cols found in sidecars\", flush=True)\n\n    # g-xTB-consistent BDE/BDFE sidecars (both sides), the 4 new candidate features\n    prod_bde = pd.read_csv(f\"{H}/products_bdfe_gxtb_descriptors.csv\",\n                            usecols=[\"id\", \"bdfe_gxtb_kcal\", \"bde_gxtb_kcal\"]).rename(\n        columns={\"bdfe_gxtb_kcal\": \"prod_bdfe_gxtb_kcal\", \"bde_gxtb_kcal\": \"prod_bde_gxtb_kcal\"})\n    ald_bde = pd.read_csv(f\"{H}/aldehydes_bdfe_gxtb_descriptors.csv\",\n                           usecols=[\"id\", \"bdfe_gxtb_kcal\", \"bde_gxtb_kcal\"]).rename(\n        columns={\"id\": \"ald_id\", \"bdfe_gxtb_kcal\": \"ald_bdfe_gxtb_kcal\", \"bde_gxtb_kcal\": \"ald_bde_gxtb_kcal\"})\n    # sanity filter: non-converged-SCF garbage (|value|>200 kcal/mol impossible for a real bond)\n    for c in [\"prod_bdfe_gxtb_kcal\", \"prod_bde_gxtb_kcal\"]:\n        prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan\n    for c in [\"ald_bdfe_gxtb_kcal\", \"ald_bde_gxtb_kcal\"]:\n        ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan\n\n    full = p.copy(); full[\"ald_id\"] = full[\"donor_id\"].astype(\"Int64\"); full = full.merge(a_r, on=\"ald_id\", how=\"left\")\n    full = add_global(full, \"smiles\")\n    full = full.merge(prod_mrd[[\"id\"] + prod_mrd_cols], on=\"id\", how=\"left\")\n    full = full.merge(ald_mrd[[\"ald_id\"] + ald_mrd_cols], on=\"ald_id\", how=\"left\")\n    full = full.merge(prod_bde, on=\"id\", how=\"left\")\n    full = full.merge(ald_bde, on=\"ald_id\", how=\"left\")\n\n    FEATS = FEATS_72 + prod_mrd_cols + ald_mrd_cols + BDE_COLS\n    print(f\"total features: {len(FEATS)} (72 champion + {len(prod_mrd_cols)+len(ald_mrd_cols)} mordred + {len(BDE_COLS)} g-xTB BDE/BDFE)\", flush=True)\n\n    df = full.merge(dft, on=\"id\").merge(cls, on=\"id\", how=\"left\")\n    # only require the core 72 + labels complete; mordred + BDE/BDFE sparse blocks are\n    # median-imputed (fit on train only) rather than dropped, per descriptor-search-exhausted's\n    # footgun lesson (union-of-NaN over ~1800+ sparse cols would leave ~0 rows otherwise)\n    df = df.dropna(subset=[\"dG_gxtb_kcal\", \"dG_orca_kcal\"] + FEATS_72).reset_index(drop=True)\n    df = df[df[\"dG_orca_kcal\"].abs() < 60].reset_index(drop=True)\n    df[\"delta\"] = df[\"dG_orca_kcal\"] - df[\"dG_gxtb_kcal\"]\n    print(f\"labeled rows: {len(df):,}\", flush=True)\n    print(f\"g-xTB BDE/BDFE coverage: prod_bde {df['prod_bde_gxtb_kcal'].notna().mean()*100:.1f}% \"\n          f\"ald_bde {df['ald_bde_gxtb_kcal'].notna().mean()*100:.1f}% \"\n          f\"prod_bdfe {df['prod_bdfe_gxtb_kcal'].notna().mean()*100:.1f}% \"\n          f\"ald_bdfe {df['ald_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%\", flush=True)\n\n    rng = np.random.default_rng(42); idx = rng.permutation(len(df))\n    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]\n    imp = SimpleImputer(strategy=\"median\").fit(df[FEATS].values[tr])\n    Xtr_raw, Xva_raw, Xte_raw = imp.transform(df[FEATS].values[tr]), imp.transform(df[FEATS].values[va]), imp.transform(df[FEATS].values[te])\n    sc = StandardScaler().fit(Xtr_raw)\n    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)\n    dtr, dva = df.delta.values[tr], df.delta.values[va]\n    gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]\n\n    # ── train members, keeping training-history curves ──────────────────────\n    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,\n                       early_stopping=True, n_iter_no_change=12, validation_fraction=0.1)\n    mlp.fit(Xtr, dtr)\n    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)\n    xgb8.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)\n    xgb10.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)\n    members = [(\"MLP\", mlp), (\"XGB_d8\", xgb8), (\"XGB_d10\", xgb10)]\n\n    preds_te = [m.predict(Xte) for _, m in members]\n    pred = np.vstack(preds_te).mean(0)\n\n    quant = make_quantiles()\n    for q, m in quant.items(): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)\n    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)\n\n    yhat = gte + pred; err = np.abs(yhat - yte)\n    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))\n    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())\n    thr = float(np.quantile(unc, 1 - ROUTE_FRAC))\n    conf = unc < thr\n    print(f\"{NAME} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\", flush=True)\n    print(f\"confident {conf.mean()*100:.0f}% MAE={err[conf].mean():.3f}  routed MAE={err[~conf].mean():.3f}\", flush=True)\n\n    # ── save production bundle ───────────────────────────────────────────\n    bundle = MODELDIR / f\"gxtb_dft_correction_{NAME}_{TAG}.joblib\"\n    joblib.dump({\"members\": members, \"quantiles\": quant, \"imputer\": imp, \"scaler\": sc,\n                \"features\": FEATS, \"route_width_threshold\": thr, \"route_frac\": ROUTE_FRAC,\n                \"target\": \"DFT_r2scan3c - gxtb (kcal/mol)\", \"test_mae\": mae, \"n_train\": len(tr)}, bundle)\n    print(\"wrote\", bundle, flush=True)\n\n    # ── full-library prediction ──────────────────────────────────────────\n    fl = full.dropna(subset=[\"dG_gxtb_kcal\"] + FEATS_72).copy()\n    Xf = sc.transform(imp.transform(fl[FEATS].values))\n    fl[\"delta_pred\"] = np.vstack([m.predict(Xf) for _, m in members]).mean(0)\n    fl[\"uncertainty_pi_width\"] = quant[0.95].predict(Xf) - quant[0.05].predict(Xf)\n    fl[\"dG_gxtb_corrected_final\"] = fl[\"dG_gxtb_kcal\"] + fl[\"delta_pred\"]\n    fl[\"route_to_dft\"] = fl[\"uncertainty_pi_width\"] >= thr\n    fl[[\"id\", \"smiles\", \"dG_gxtb_kcal\", \"dG_gxtb_corrected_final\", \"uncertainty_pi_width\", \"route_to_dft\"]].to_csv(\n        OUT / f\"products_dG_corrected_{NAME}_{TAG}.csv\", index=False)\n    print(f\"wrote full-library predictions ({len(fl):,} mols)\", flush=True)\n\n    # ══ diagnostics (same shape as finalize_correction_mordred_slim.py, 100-series to avoid\n    # clobbering the 90-99 mordredslim271 series) ═══════════════════════════\n    fig, ax = plt.subplots(figsize=(6, 6))\n    ax.scatter(yte, yhat, s=4, alpha=0.25, color=\"#2171b5\")\n    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())\n    ax.plot([lo, hi], [lo, hi], \"k--\", lw=1)\n    ax.set_xlabel(\"DFT dG_orca (kcal/mol)\"); ax.set_ylabel(\"corrected prediction (kcal/mol)\")\n    ax.set_title(f\"{NAME} parity (test, n={len(yte):,})\\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\")\n    savefig(f\"100_parity_{NAME}.png\")\n\n    resid = yhat - yte\n    fig, ax = plt.subplots(figsize=(7, 5))\n    ax.hist(resid, bins=80, color=\"#6baed6\", edgecolor=\"none\")\n    ax.axvline(0, color=\"k\", lw=1)\n    ax.set_xlabel(\"prediction - DFT (kcal/mol)\"); ax.set_ylabel(\"count\")\n    ax.set_title(f\"{NAME} residual distribution\\nmean={resid.mean():.3f} std={resid.std():.3f}\")\n    savefig(f\"101_residual_hist_{NAME}.png\")\n\n    fig, ax = plt.subplots(figsize=(7, 5))\n    ax.scatter(yhat, resid, s=4, alpha=0.25, color=\"#cb181d\")\n    ax.axhline(0, color=\"k\", lw=1)\n    ax.set_xlabel(\"predicted dG (kcal/mol)\"); ax.set_ylabel(\"residual (pred - DFT)\")\n    ax.set_title(f\"{NAME} residual vs predicted\")\n    savefig(f\"102_residual_vs_pred_{NAME}.png\")\n\n    fig, ax = plt.subplots(figsize=(7, 5))\n    ax.plot(mlp.loss_curve_, label=\"train loss\")\n    if hasattr(mlp, \"validation_scores_\") and mlp.validation_scores_:\n        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color=\"#cb181d\", label=\"val R2\")\n        ax2.set_ylabel(\"validation R2\", color=\"#cb181d\")\n    ax.set_xlabel(\"epoch\"); ax.set_ylabel(\"training loss\")\n    ax.set_title(f\"{NAME} MLP training curve (stopped at epoch {len(mlp.loss_curve_)})\")\n    savefig(f\"103_mlp_loss_curve_{NAME}.png\")\n\n    for m, name, fname in [(xgb8, \"XGB_d8\", f\"104_xgb_d8_curve_{NAME}.png\"),\n                           (xgb10, \"XGB_d10\", f\"105_xgb_d10_curve_{NAME}.png\")]:\n        ev = m.evals_result()\n        fig, ax = plt.subplots(figsize=(7, 5))\n        ax.plot(ev[\"validation_0\"][\"mae\"], label=\"train MAE\"); ax.plot(ev[\"validation_1\"][\"mae\"], label=\"val MAE\")\n        ax.axvline(m.best_iteration, color=\"k\", ls=\"--\", lw=1, label=f\"best_iter={m.best_iteration}\")\n        ax.set_xlabel(\"boosting round\"); ax.set_ylabel(\"MAE (kcal/mol)\")\n        ax.set_title(f\"{NAME} {name} training curve\"); ax.legend()\n        savefig(fname)\n\n    err_thr = float(np.quantile(err, 1 - ROUTE_FRAC))\n    y_true_bin = (err >= err_thr).astype(int)\n    fpr, tpr, _ = roc_curve(y_true_bin, unc); roc_auc = auc(fpr, tpr)\n    fig, ax = plt.subplots(figsize=(6, 6))\n    ax.plot(fpr, tpr, color=\"#2171b5\", lw=2, label=f\"AUC={roc_auc:.3f}\")\n    ax.plot([0, 1], [0, 1], \"k--\", lw=1)\n    ax.set_xlabel(\"false positive rate\"); ax.set_ylabel(\"true positive rate\")\n    ax.set_title(f\"{NAME} uncertainty-routing ROC\\n(worst {ROUTE_FRAC*100:.0f}% by true error)\")\n    ax.legend(loc=\"lower right\")\n    savefig(f\"106_uncertainty_roc_{NAME}.png\")\n\n    fig, ax = plt.subplots(figsize=(7, 5))\n    ax.scatter(unc[conf], err[conf], s=4, alpha=0.3, color=\"#2171b5\", label=f\"confident ({conf.mean()*100:.0f}%)\")\n    ax.scatter(unc[~conf], err[~conf], s=4, alpha=0.3, color=\"#cb181d\", label=f\"routed ({(~conf).mean()*100:.0f}%)\")\n    ax.axvline(thr, color=\"k\", ls=\"--\", lw=1)\n    ax.set_xlabel(\"quantile PI width (uncertainty)\"); ax.set_ylabel(\"|error| (kcal/mol)\")\n    ax.set_title(f\"{NAME} error vs uncertainty (conf MAE={err[conf].mean():.3f}, routed MAE={err[~conf].mean():.3f})\")\n    ax.legend()\n    savefig(f\"107_error_vs_uncertainty_{NAME}.png\")\n\n    molwt = df[\"g_MolWt\"].values[te]; scope_te = df[\"cls\"].values[te]\n    fig, ax = plt.subplots(figsize=(7, 5))\n    for s, c in [(\"aromatic\", \"#2171b5\"), (\"aliphatic\", \"#cb181d\")]:\n        mk = scope_te == s\n        ax.scatter(molwt[mk], err[mk], s=4, alpha=0.3, color=c, label=s)\n    ax.set_xlabel(\"product MolWt (g/mol)\"); ax.set_ylabel(\"|error| (kcal/mol)\")\n    ax.set_title(f\"{NAME} error vs molecular size\"); ax.legend()\n    savefig(f\"108_error_vs_molwt_{NAME}.png\")\n\n    te_df = df.iloc[te].reset_index(drop=True).copy()\n    te_df[\"dG_pred\"] = yhat; te_df[\"error\"] = err; te_df[\"uncertainty_pi_width\"] = unc\n    te_df[\"route_to_dft\"] = ~conf\n    keep = [\"id\", \"ald_id\", \"smiles\", \"ald_smiles\", \"cls\", \"dG_orca_kcal\", \"dG_gxtb_kcal\",\n            \"dG_pred\", \"error\", \"uncertainty_pi_width\", \"route_to_dft\"] + BDE_COLS\n    te_df[keep].to_csv(OUT / f\"test_predictions_{NAME}_{TAG}.csv\", index=False)\n\n    worst = te_df.sort_values(\"error\", ascending=False).head(50).reset_index(drop=True)\n    worst[keep].to_csv(OUT / f\"worst_mispredictions_{NAME}_{TAG}.csv\", index=False)\n    top20 = worst.head(20)\n    mols, legends = [], []\n    for _, r in top20.iterrows():\n        m = Chem.MolFromSmiles(str(r[\"ald_smiles\"]))\n        if m is None: continue\n        mols.append(m)\n        legends.append(f\"id={r['id']} err={r['error']:.2f}\\npred={r['dG_pred']:.1f} true={r['dG_orca_kcal']:.1f}\")\n    if mols:\n        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 220), legends=legends)\n        img.save(OUT / f\"109_worst_mispredicted_aldehydes_{NAME}.png\")\n        print(f\"wrote 109_worst_mispredicted_aldehydes_{NAME}.png\", flush=True)\n\n    rep = OUT / f\"REPORT_{NAME}_FINAL_{TAG}.md\"\n    with open(rep, \"w\") as fh:\n        fh.write(f\"# {NAME} production model ({TAG})\\n\\n\")\n        fh.write(f\"- Features: **{len(FEATS)}** (mordredslim271's 271 [72 champion QM + \"\n                 f\"{len(prod_mrd_cols)+len(ald_mrd_cols)} targeted mordred] + 4 g-xTB-consistent \"\n                 f\"BDE/BDFE: {', '.join(BDE_COLS)})\\n\")\n        fh.write(f\"- **TEST MAE {mae:.3f}, RMSE {rmse:.3f}, R2 {r2:.3f}** vs mordredslim271 production \"\n                 f\"champion (test MAE 1.525) and the quick 2-member XGB check that motivated this run \"\n                 f\"(1.612 -> 1.563, REPORT_bdfe_gxtb_full_augment_20260706.md)\\n\")\n        fh.write(f\"- Uncertainty routing: confident {conf.mean()*100:.0f}% MAE {err[conf].mean():.3f} | \"\n                 f\"routed MAE {err[~conf].mean():.3f} | ROC AUC {roc_auc:.3f}\\n\")\n        fh.write(f\"- Model: `{bundle}`\\n- Full-library output: `products_dG_corrected_{NAME}_{TAG}.csv` \"\n                 f\"({len(fl):,} mols; {int(fl.route_to_dft.sum()):,} flagged route_to_dft)\\n\")\n    print(\"wrote\", rep, flush=True)\n    print(\"DONE\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "bb470146ab53931fe9b42464c9a0e917997a55ae",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>   "display_title": "finalize_correction_mordredslim271_bdegxtb.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "prod_bdfe_smiles",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 20
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "path": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_title": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"Pure-SMILES (no-xtb) surrogate for product C-C BDFE -- the fast/no-quantum tier, mirroring\nthis project's established 2D-surrogate pattern (pipeline/train_surrogate.py, which does the\nsame for the main reaction dG). The QM-feature surrogate (train_bde_surrogate.py) already\ngets R2=0.901 for product BDFE using 72-champion QM + mordred feats -- but those need an xtb\ngeometry+SP to obtain, so they aren't \"free\" for screening brand-new molecules. This script\ntests how much of that predictive power survives using ONLY RDKit 2D descriptors computable\ndirectly from the SMILES string (no conformer, no xtb at all) -- the real no-quantum-cost\ncomparison point.\n\nUses the GFN2-level product BDFE (products_bdfe2_descriptors.csv, full library, 96% filled)\nas the training target since it's the largest available label set right now.\n\"\"\"\nimport sys\nimport time\nfrom pathlib import Path\nimport numpy as np, pandas as pd, joblib\nimport matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom xgboost import XGBRegressor\n\nsys.path.insert(0, str(Path(__file__).resolve().parent.parent / \"compute\"))\nfrom ald_descriptors import calc_rdkit, _RDKIT_FIELDS\n\nR = \"/scratch-shared/schen3/benzoin-dg\"; H = f\"{R}/data/cross_benzoin/homo_v6\"\nOUT = Path(f\"{H}/viz_gxtb_20260625\"); OUT.mkdir(exist_ok=True)\nTAG = time.strftime(\"%Y%m%d\"); MODELDIR = Path(f\"{R}/pipeline/models\"); MODELDIR.mkdir(exist_ok=True)\n\n\ndef _xgb(depth, ne):\n    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,\n                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,\n                        early_stopping_rounds=60, eval_metric=\"mae\")\n\n\ndef savefig(name):\n    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches=\"tight\"); plt.close()\n    print(\"wrote\", name, flush=True)\n\n\ndef main():\n    p = pd.read_csv(f\"{H}/products_all.csv\", usecols=[\"id\", \"smiles\"], low_memory=False).drop_duplicates(\"id\")\n    bdfe = pd.read_csv(f\"{H}/products_bdfe2_descriptors.csv\", usecols=[\"id\", \"bdfe_xtb_kcal\"])\n    bdfe = bdfe[bdfe[\"bdfe_xtb_kcal\"].abs() <= 200]  # drop pathological SCF-failure rows\n\n    df = p.merge(bdfe, on=\"id\", how=\"inner\")\n    print(f\"n={len(df):,} products with valid BDFE label\", flush=True)\n\n    print(\"computing pure-SMILES 2D descriptors (no xtb)...\", flush=True)\n    t0 = time.time()\n    feat_rows = [calc_rdkit(s) for s in df[\"smiles\"]]\n    print(f\"done in {time.time()-t0:.1f}s\", flush=True)\n    feat_df = pd.DataFrame(feat_rows)\n    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)\n    feats = _RDKIT_FIELDS\n    for c in feats:\n        df[c] = pd.to_numeric(df[c], errors=\"coerce\")\n\n    d = df.dropna(subset=feats + [\"bdfe_xtb_kcal\"]).reset_index(drop=True)\n    print(f\"n={len(d):,} after dropna, n_feat={len(feats)}\", flush=True)\n\n    rng = np.random.default_rng(42); idx = rng.permutation(len(d))\n    ntr, nva = int(.7 * len(d)), int(.9 * len(d)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]\n    imp = SimpleImputer(strategy=\"median\").fit(d[feats].values[tr])\n    Xtr_raw, Xva_raw, Xte_raw = imp.transform(d[feats].values[tr]), imp.transform(d[feats].values[va]), imp.transform(d[feats].values[te])\n    sc = StandardScaler().fit(Xtr_raw)\n    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)\n    ytr, yva, yte = d[\"bdfe_xtb_kcal\"].values[tr], d[\"bdfe_xtb_kcal\"].values[va], d[\"bdfe_xtb_kcal\"].values[te]\n\n    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)\n    xgb8.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    xgb10.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)\n    yhat = np.vstack([xgb8.predict(Xte), xgb10.predict(Xte)]).mean(0)\n    err = np.abs(yhat - yte)\n    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))\n    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())\n    print(f\"[prod_bdfe_smiles2d] n={len(d):,} n_feat={len(feats)} test MAE={mae:.3f} \"\n         f\"RMSE={rmse:.3f} R2={r2:.3f}\", flush=True)\n\n    bundle = MODELDIR / f\"bde_surrogate_prod_bdfe_smiles2d_{TAG}.joblib\"\n    joblib.dump({\"members\": [(\"XGB_d8\", xgb8), (\"XGB_d10\", xgb10)], \"imputer\": imp,\n                \"scaler\": sc, \"features\": feats, \"target\": \"bdfe_xtb_kcal (GFN2)\",\n                \"test_mae\": mae, \"test_r2\": r2, \"n_train\": len(tr)}, bundle)\n    print(\"wrote\", bundle, flush=True)\n\n    fig, ax = plt.subplots(figsize=(6, 6))\n    ax.scatter(yte, yhat, s=4, alpha=0.25, color=\"#6a51a3\")\n    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())\n    ax.plot([lo, hi], [lo, hi], \"k--\", lw=1)\n    ax.set_xlabel(\"real xtb product BDFE (kcal/mol)\"); ax.set_ylabel(\"pure-SMILES surrogate prediction (kcal/mol)\")\n    ax.set_title(f\"product BDFE pure-SMILES(2D) surrogate parity (test, n={len(yte):,})\\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}\")\n    savefig(f\"130_parity_prod_bdfe_smiles2d_{TAG}.png\")\n\n    imp_gain = xgb8.get_booster().get_score(importance_type=\"gain\")\n    top_feats = sorted(imp_gain.items(), key=lambda kv: kv[1], reverse=True)\n    feat_names = [feats[int(k[1:])] for k, _ in top_feats]\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ax.barh(range(len(top_feats))[::-1], [v for _, v in top_feats])\n    ax.set_yticks(range(len(top_feats))[::-1]); ax.set_yticklabels(feat_names, fontsize=9)\n    ax.set_xlabel(\"XGB_d8 gain importance\")\n    ax.set_title(\"product BDFE pure-SMILES(2D) surrogate feature importance\")\n    savefig(f\"131_feat_importance_prod_bdfe_smiles2d_{TAG}.png\")\n\n    rep = OUT / f\"REPORT_prod_bdfe_smiles2d_surrogate_{TAG}.md\"\n    with open(rep, \"w\") as fh:\n        fh.write(f\"# Product BDFE pure-SMILES (no-xtb) surrogate ({TAG})\\n\\n\")\n        fh.write(f\"Predicts product C-C BDFE (GFN2, {len(d):,} molecules) from ONLY \"\n                f\"{len(feats)} RDKit 2D descriptors computable directly from the SMILES \"\n                f\"string -- no conformer, no xtb at all. Compare to the QM-feature surrogate \"\n                f\"(`train_bde_surrogate.py`, needs xtb geometry+SP): R2=0.901, MAE=2.70.\\n\\n\")\n        fh.write(f\"- **pure-SMILES(2D)**: n_feat={len(feats)} n={len(d):,} MAE={mae:.3f} \"\n                f\"RMSE={rmse:.3f} R2={r2:.3f}\\n\")\n        fh.write(f\"- Top predictive feats: {', '.join(feat_names[:5])}\\n\\n\")\n        gap = 0.901 - r2\n        verdict = (\"substantial (QM electronic-structure info is doing real work; \"\n                  \"not a cheap substitute)\" if gap > 0.15 else\n                  \"small (2D descriptors capture most of the signal; may be usable as a \"\n                  \"cheap prospective pre-filter)\")\n        fh.write(f\"## Interpretation\\n\\nR2 gap vs the QM-feature surrogate: {gap:.3f} -- \"\n                f\"**{verdict}**.\\n\")\n    print(\"wrote\", rep, flush=True)\n    print(\"DONE\", flush=True)\n\n\nif __name__ == \"__main__\":\n    main()\n",
>   "encoding": "utf-8",
>   "sha": "8bf35a7cfec594b1ce2f9ca57dd59e69e90ed794",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>   "display_title": "train_prod_bdfe_smiles_surrogate.py"
> }
> ```
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00169e`
>
> </details>
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "scaffold split OR external validation OR experimental",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 30
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": []
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "mordred_slim_selection_20260703.json",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 20
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_mordred_slim_20260703.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_mordred_slim_20260703.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_mordred_slim_20260703.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_mordred_slim_20260703.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "path": "pipeline/analysis/train_bde_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "display_title": "pipeline/analysis/train_bde_surrogate.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_bdfe_feature_correlation.py",
>       "path": "pipeline/analysis/analyze_bdfe_feature_correlation.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_bdfe_feature_correlation.py",
>       "display_title": "pipeline/analysis/analyze_bdfe_feature_correlation.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bde.py",
>       "path": "pipeline/analysis/finalize_correction_bde.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bde.py",
>       "display_title": "pipeline/analysis/finalize_correction_bde.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_subset_hypervalent_champion275.py",
>       "path": "pipeline/analysis/shap_subset_hypervalent_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_subset_hypervalent_champion275.py",
>       "display_title": "pipeline/analysis/shap_subset_hypervalent_champion275.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_ensemble_reeval_matched.py",
>       "path": "pipeline/analysis/gnn_ensemble_reeval_matched.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_ensemble_reeval_matched.py",
>       "display_title": "pipeline/analysis/gnn_ensemble_reeval_matched.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_aligned_v2.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275_aligned_v2.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_aligned_v2.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275_aligned_v2.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_aligned.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275_aligned.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_aligned.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275_aligned.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/promote_gnn_stacking_full_library.py",
>       "path": "pipeline/analysis/promote_gnn_stacking_full_library.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/promote_gnn_stacking_full_library.py",
>       "display_title": "pipeline/analysis/promote_gnn_stacking_full_library.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_ensemble.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275_ensemble.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275_ensemble.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275_ensemble.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "path": "pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py",
>       "display_title": "pipeline/analysis/shap_slim_mordredslim271_bdegxtb.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "path": "pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py",
>       "display_title": "pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_hypervalent_tags.py",
>       "path": "pipeline/analysis/finalize_correction_hypervalent_tags.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_hypervalent_tags.py",
>       "display_title": "pipeline/analysis/finalize_correction_hypervalent_tags.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordred_slim.py",
>       "path": "pipeline/analysis/finalize_correction_mordred_slim.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_mordred_slim.py",
>       "display_title": "pipeline/analysis/finalize_correction_mordred_slim.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_aldonly.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_aldonly.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_aldonly.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_aldonly.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275.py"
>     }
>   ]
> }
> ```
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `"acs.jctc.3c00710" MAE BDE MACE ALFABET GFN2 ...`
>
> </details>
>
> </details>

</details>

先给结论：这个仓库的 BDE/BDFE 模块与 ALFABET 并不是同一种模型，不能直接拿 MAE 横向排名。

- 仓库的核心任务是预测 NHC 催化 homo-benzoin 反应的溶液相 ΔG；BDE/BDFE 只是四个反应专用描述符。
- ALFABET 是从 SMILES 直接预测任意指定键均裂 BDE/BDFE 的通用监督式 GNN。
- 你给出的 2026 Nature Communications 工作主要研究通用 ML 势能模型是否“自发学会”键能概念，并不是一个精确的逐分子 BDE 预测器。
- 就现有证据而言，仓库中真正有价值的是 g-xTB BDE，而不是昂贵的 BDFE；但它对最终 ΔG 的增益只有 0.022 kcal/mol，仍需更严格的 OOD 和统计验证。

另外，正确名称是 **ALFABET**，不是 ALPHABET：A machine-Learning derived, Fast, Accurate Bond dissociation Enthalpy Tool。

## 1. 三者的核心差异

| 维度               | ElioChen 仓库                                                  | ALFABET/BDE-db2                | Hattori 2026 E3D/Allegro           |
| ------------------ | -------------------------------------------------------------- | ------------------------------ | ---------------------------------- |
| 主要目标           | 苯偶姻缩合反应 ΔG                                             | 任意指定键的均裂 BDE/BDFE      | 解释 MLIP 内部是否涌现键能概念     |
| BDE 在系统中的角色 | 反应 ΔG 模型的输入描述符                                      | 最终预测目标                   | 从边能量分解中提取的内部表征       |
| 键类型             | 醛 C–H、产物新生 C–C 两类固定键                              | 多种非环外单键                 | C–H/C–C/C–O/C–N 等键型统计分布 |
| 输入               | 3D 几何、xTB/Multiwfn、Mordred、g-xTB ΔG 等                   | 2D SMILES/RDKit 图             | 3D 原子坐标与元素                  |
| 模型               | g-xTB/GFN2 计算 + XGB/MLP Δ-learning；另有 XGB 键能 surrogate | 六层 message-passing GNN       | E(3) 等变 Allegro MLIP + E3D 分解  |
| 参考层级           | g-xTB 电子能 + GFN2 RRHO，DMSO                                 | M06-2X/def2-TZVP，298 K、1 atm | SPICE2/MatPES 的系统能量和力       |
| 是否直接预测键能   | surrogate 是；生产 ΔG 模型不是                                | 是                             | 严格说不是                         |
| 推理成本           | 生产特征仍需结构与 xTB/g-xTB；surrogate 较快                   | 仅 SMILES，秒级/亚秒级         | 需 3D 结构和 MLIP 前向计算         |
| 适用范围           | 很窄但反应专用                                                 | 广泛有机均裂                   | MLIP 表征、势能面和反应研究        |

## 2. 仓库模型究竟在做什么

仓库计算两个特定反应位点：

1. 醛底物的 formyl C–H 均裂；
2. benzoin 产物中新生成 C–C 键的均裂。

它采用的定义是：

\[
\mathrm{BDE}_{g-xTB}
=E_{g-xTB}(A^\bullet)+E_{g-xTB}(B^\bullet)-E_{g-xTB}(AB)
\]

\[
G_{g-xTB}=E_{g-xTB}+
\left(G_{\mathrm{GFN2}}-E_{\mathrm{GFN2}}\right)
\]

\[
\mathrm{BDFE}_{g-xTB}
=G_{g-xTB}(A^\bullet)+G_{g-xTB}(B^\bullet)-G_{g-xTB}(AB)
\]

片段先在 GFN2-xTB/ALPB(DMSO) 下优化并计算 Hessian，再在相同几何上做 g-xTB/COSMO(DMSO) 单点。因此：

- 仓库所谓的 BDE 更接近“g-xTB 电子解离能”，不是 ALFABET 的 298 K、含 ZPE 的气相键解离焓；
- BDFE 是 g-xTB 电子能与 GFN2 RRHO 热校正的混合量；
- 两者的参考态、溶剂环境、电子结构层级都不同。

详见仓库的[中文方法说明](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/docs/METHODS_BDE_gxtb_20260707_zh.md)和[计算代码](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/compute/calc_bde_free_energy_gxtb.py)。

### 作为 ΔG 描述符的效果

仓库最终生产模型包含：

- 72 个 QM/局部/立体描述符；
- 199 个筛选后的 Mordred 描述符；
- 4 个 g-xTB BDE/BDFE 描述符；
- MLP、XGB-depth8、XGB-depth10 集成；
- 分位数模型作为不确定度路由。

随机 70:20:10 划分上的结果为：

| ΔG 模型                    |       测试 MAE |
| --------------------------- | -------------: |
| 271 特征，不含 BDE/BDFE     | 1.525 kcal/mol |
| 275 特征，加入四个 BDE/BDFE | 1.503 kcal/mol |
| 改善                        | 0.022 kcal/mol |

完整指标为 MAE 1.503、RMSE 2.257、R² 0.875；置信度最高的 85% 样本 MAE 1.252。见[最终模型报告](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md)。

这个结果说明 BDE/BDFE 有补充信息，但效应很小。仓库自己的快速消融得到：

| 特征增量    | ΔG 测试 MAE |              相对基线 |
| ----------- | -----------: | --------------------: |
| 无 BDE/BDFE |        1.612 |                    — |
| + BDFE      |        1.605 | −0.007，基本为空结果 |
| + BDE       |        1.580 |               −0.032 |
| + BDE+BDFE  |        1.563 |               −0.049 |

SHAP 中两个 BDE 分别排第 4 和第 6，而两个 BDFE 只排第 15 和第 38；BDE 总重要性约为 BDFE 的 3.7 倍。因此仓库关于“新分子只计算 BDE、跳过 Hessian/BDFE”的建议是合理的。

### 仓库真正的 BDE/BDFE surrogate

仓库还训练了用现有 QM+Mordred 描述符重建上述 g-xTB 键能的 XGB surrogate：

| 目标           |       测试 MAE |   R² |
| -------------- | -------------: | ----: |
| 醛 C–H BDE    | 3.027 kcal/mol | 0.094 |
| 醛 C–H BDFE   |          2.018 | 0.797 |
| 产物 C–C BDE  |          3.878 | 0.652 |
| 产物 C–C BDFE |          2.704 | 0.901 |

见[surrogate 报告](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_bde_surrogate_20260704.md)。

这里有一个有趣但合理的现象：

- BDFE 更容易被其他描述符重建，说明它与已有电子/结构描述符高度冗余；
- 醛 C–H BDE 的 R² 仅 0.094，说明它携带独立信息；
- 独立信息并不等于能被 surrogate 精确预测，所以真实 g-xTB BDE 对 ΔG 模型更有价值。

## 3. 与 ALFABET 的比较

2020 年原始 ALFABET 在 42,577 个小分子、290,664 个 BDE 上训练，针对 C/H/N/O 中性闭壳层分子的非环单键均裂，对 M06-2X/def2-TZVP 的测试 MAE 为 0.58 kcal/mol。[原始论文](https://www.nature.com/articles/s41467-020-16201-z)

2023 年 BDE-db2 扩展版更适合作为当前比较对象：

- 531,244 个唯一 homolytic BDE/BDFE；
- 元素覆盖 C、H、N、O、S、F、Cl、P、Br、I；
- M06-2X/def2-TZVP；
- BDE 包含未缩放 ZPE；
- BDFE 为 298 K、1 atm 的 RRHO Gibbs 自由能；
- 直接从 2D SMILES 预测，不需要几何或 QM 描述符；
- 分子级独立测试集包含 1,000 个分子、8,084 个解离反应。

其内部测试结果为：

| 目标     |    MAE vs DFT |
| -------- | ------------: |
| BDE      | 0.61 kcal/mol |
| BDFE     | 0.60 kcal/mol |
| C–C BDE | 0.77 kcal/mol |
| C–H BDE | 0.74 kcal/mol |

但它在真正 OOD 化学空间上的表现明显下降：

- 初始 haloheterocycle 外部集：BDE/BDFE MAE 5.67/5.74；
- 针对性加入少量相似结构后：约 0.84/0.83；
- polyhaloalkyl 最终约 1.19/1.20；
- 4 个大型卤代天然产物：约 1.76；
- 与实验卤代烷 BDE 比较：MAE 2.12；
- PFAS 外部集：BDE 1.15、BDFE 1.51 kcal/mol。

这非常重要：0.6 kcal/mol 是同一 DFT oracle、训练域附近的分子级测试结果，不代表对任意新化学空间都有 0.6 的精度。[BDE-db2/扩展 ALFABET 论文](https://pubs.rsc.org/en/content/articlepdf/2023/dd/d3dd00169e)

### 谁更强？

如果问题是“从 SMILES 快速预测标准气相 homolytic BDE/BDFE”，ALFABET 明显更成熟：

- 训练目标定义标准；
- 数据覆盖广；
- 仅需 2D SMILES；
- 内部误差约 0.6–0.8 kcal/mol；
- 有多个外部验证集。

如果问题是“预测 DMSO 中苯偶姻缩合 ΔG”，仓库模型更相关：

- 它针对固定反应和固定断键；
- 描述符与 g-xTB ΔG 基线在同一方法体系中；
- 直接优化最终反应 ΔG，而非单独追求通用键能。

因此不能说 ALFABET 的 0.60 优于仓库 ΔG 模型的 1.503：两者预测的是不同物理量。能够比较的是 ALFABET 的 C–H/C–C BDE 与仓库 BDE surrogate，但也必须先统一参考态和标签。

## 4. 2026 Nature Communications 论文应如何定位

Hattori 等人的 2026 工作使用：

- E(3)-equivariant Allegro；
- SPICE2 稳定分子数据训练；
- 总能量、力等常规 MLIP 目标；
- 无显式 BDE 标签；
- 新提出的 edge-wise emergent energy decomposition，E3D；
- 对每条有向边的能量 \(\epsilon_{ij}\) 构造
  \(D_{ij}=\epsilon_{ij}+\epsilon_{ji}\) 和
  \(A_{ij}=\epsilon_{ij}-\epsilon_{ji}\)。

模型确实学到了合理的：

- C–H、N–H、O–H 键强度顺序；
- 单键、双键、三键顺序；
- 跨 SPICE2 和 MatPES 数据集一致的键型表征。

但论文明确指出，emergent BDE 与文献值平均仍相差约 **0.3 eV，约 6.9 kcal/mol**；同期监督式 BonDNet 误差约 0.022 eV。因此它当前不是 ALFABET 的精度竞争者，而是：

1. MLIP 可解释性方法；
2. 判断数据规模何时触及“scaling wall”的诊断工具；
3. 研究稳定结构训练如何产生反应和过渡态能力的框架。

其主要预测改进是过渡态任务：SPICE2+MatPES 混合训练把 activation-energy MAE 从 0.58 eV 降到 0.44 eV，但该误差仍远大于高精度反应筛选通常需要的范围。[Nature Communications 论文](https://www.nature.com/articles/s41467-026-74919-8)，[全文 PDF](https://www.nature.com/articles/s41467-026-74919-8_reference.pdf)

所以，这篇论文对仓库最有价值的不是替换其 BDE 模型，而是提供两个新方向：

- 从 ΔG 模型或未来 3D GNN 的边表示中提取“新生 C–C”和“醛 C–H”的显式边能量；
- 用边能量分布、非对称性和信息熵检查模型到底学到了键能，还是只学到了分子大小、官能团和结构共现。

## 5. 截至 2026-07-14 的主要研究趋势

### 直接监督式键能模型

ALFABET/BDE-db2 仍是通用、快速、可直接部署的强基线。它的关键经验不是盲目增大模型，而是使用外部集识别化学空间空洞，再用少量定向数据补齐。

### MLIP 通过能量差计算 BDE

2024 年的 MACE 工作不直接回归 BDE，而是分别预测母体和自由基能量，再取差值。在 CYP 3A4 脂肪族 C–H 数据集上得到 BDE RMSE 1.37 kcal/mol，并在位点排序上优于 ALFABET、AM1 和 GFN2-xTB。它还能产生平滑的拉键势能曲线和近似反应路径，这是 ALFABET 不具备的。[MACE BDE 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10782450/)

### 用物理约束补足断键数据

2025 年的 physics-constrained data augmentation 使用 Morse 势生成拉键路径上的辅助数据，改善 ANI-1xnr 的解离曲线和 BDE，同时尽量保持反应 MD 稳定性。它说明仅靠平衡结构或更多普通数据不一定能解决断键外推。[PCDA 论文](https://pubmed.ncbi.nlm.nih.gov/39874212/)

### 显式建模长程分子内相互作用

2026 年 OLED 研究使用 through-bond 和 through-space 双注意力，并加入色散矩阵表示几何和长程非共价作用；C–N BDE MAE 低于 0.05 eV，并通过迁移学习扩展到 C–P/C–S。这对仓库中存在氢键、构象和色散效应的 benzoin 产物尤其相关。[Chemistry of Materials 论文](https://pubs.acs.org/doi/10.1021/acs.chemmater.6c01241)

整体趋势已经从“只预测一个 BDE 数值”分化为：

- 2D 图模型：快速、大规模筛选；
- 3D MLIP：能量、力、拉键曲线和反应路径；
- 物理增强：修复断键外推；
- 长程注意力：捕获局部图网络遗漏的构象和非共价作用；
- E3D 等方法：解释模型内部是否真正形成了键能概念。

## 6. 对该仓库最合理的下一步

我建议不要立即换成 ALFABET，而是建立一个严格的共享基准。

### 第一层：键能本身的 apples-to-apples 比较

在同一批醛/benzoin 分子上比较：

1. g-xTB/GFN2 当前值；
2. 仓库 XGB surrogate；
3. ALFABET/BDE-db2；
4. MACE 或其他包含自由基的 MLIP；
5. 一小批高层 DFT 参考值。

必须分别报告：

- 醛 C–H；
- 产物 C–C；
- BDE 与 BDFE；
- 芳香族与脂肪族；
- S/P/卤素/酰胺/亚胺等困难官能团；
- MAE、RMSE、Spearman/Kendall 排序、异常值比例。

### 第二层：最终 ΔG 的消融

保持完全相同的数据划分和模型架构，仅替换四个键能特征：

| 实验 | BDE/BDFE 来源               |
| ---- | --------------------------- |
| A    | 无键能特征                  |
| B    | 仓库真实 g-xTB BDE          |
| C    | 仓库 surrogate              |
| D    | ALFABET                     |
| E    | MACE/MLIP                   |
| F    | g-xTB BDE + E3D/MLIP 边表征 |

这才能回答“哪个 BDE 模型对 benzoin ΔG 最有用”，而不仅是“哪个更接近自己的 DFT oracle”。

### 第三层：改进验证协议

仓库当前生产脚本使用 `np.random.default_rng(42).permutation` 做单次随机 70:20:10 划分。对于 22 万个相关候选结构，这可能高估对新骨架的泛化能力。见[训练脚本](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/analysis/finalize_correction_mordredslim271_bdegxtb.py)。

至少应增加：

- Bemis–Murcko scaffold split；
- 官能团 OOD split；
- 脂肪族/芳香族互相外推；
- 时间或来源拆分；
- 5–10 个随机种子的配对 bootstrap 置信区间；
- 小规模实验或高层 DFT 外部集。

尤其是 1.525 → 1.503 的 0.022 kcal/mol 改善非常小；在完成多种子配对检验前，应表述为“轻微、可能真实的增益”，而不是已经充分确立的模型优势。

## 最终判断

- **通用 BDE/BDFE 快速预测：**扩展 ALFABET 是目前更成熟的直接基线。
- **苯偶姻 ΔG 专用预测：**仓库方案更贴合任务，但 BDE/BDFE 只是辅助描述符。
- **仓库的最佳物理特征：**保留 g-xTB BDE；新分子阶段可优先不计算昂贵 BDFE。
- **2026 E3D 论文：**适合用于解释和改造未来 3D/MLIP 模型，不适合作为当前 BDE 数值预测器直接替代 ALFABET。
- **最需要补足的证据：**统一参考态的共享测试集、scaffold/OOD 验证、实验或高层 DFT 外部验证，以及 0.022 kcal/mol 增益的统计显著性。

> 我希望把醛或者benzoin产物的BDE预测发展为子课题，学习目前最先进的方法（不只是我复制的链接），模型，架构，并且与alpbet进行对比。还得考虑homo/cross benzoin的描述符问题（若使用）

<details><summary>2 previous messages</summary>

> 我会把它按一个可发表的独立子课题来设计：先定义可比较的 BDE/BDFE 任务与数据集，再系统评估 2D 键中心模型、3D 等变模型、Δ-learning 和反应型 MLIP，并单独处理 homo/cross-benzoin 中“一个产物对应两个不同醛来源”的描述符与数据划分问题。接下来我会先核对仓库里的 cross-benzoin 数据结构，再补齐 2024–2026 的主流与前沿方法。
>
> <details><summary>Ran a command, called 9 tools, searched the web</summary>
>
> - Ran `Get-Content -Raw 'C:\Users\shuai\.codex\plugins\cache\openai-curated-remote\github\0.1.8-2841cf9749ae\skills\github\SKILL.md'`
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "cross benzoin descriptor",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 25
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_rdkit_descriptors.sh",
>       "path": "cross_benzoin/slurm/submit_rdkit_descriptors.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_rdkit_descriptors.sh",
>       "display_title": "cross_benzoin/slurm/submit_rdkit_descriptors.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_rdkit_descriptors.py",
>       "path": "cross_benzoin/analysis/add_rdkit_descriptors.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_rdkit_descriptors.py",
>       "display_title": "cross_benzoin/analysis/add_rdkit_descriptors.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "path": "cross_benzoin/ARCHITECTURE.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "display_title": "cross_benzoin/ARCHITECTURE.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "path": "cross_benzoin/cb_featurize.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "display_title": "cross_benzoin/cb_featurize.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_morfeus_descriptors.sh",
>       "path": "cross_benzoin/slurm/submit_morfeus_descriptors.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_morfeus_descriptors.sh",
>       "display_title": "cross_benzoin/slurm/submit_morfeus_descriptors.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_mordred_descriptors.py",
>       "path": "cross_benzoin/analysis/add_mordred_descriptors.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_mordred_descriptors.py",
>       "display_title": "cross_benzoin/analysis/add_mordred_descriptors.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_mordred_array.sh",
>       "path": "cross_benzoin/slurm/submit_mordred_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_mordred_array.sh",
>       "display_title": "cross_benzoin/slurm/submit_mordred_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MASTER_gxtb_dft_correction_20260625.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MASTER_gxtb_dft_correction_20260625.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MASTER_gxtb_dft_correction_20260625.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_MASTER_gxtb_dft_correction_20260625.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_gxtb_products_full_20260625.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_gxtb_products_full_20260625.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_gxtb_products_full_20260625.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_gxtb_products_full_20260625.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md",
>       "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md",
>       "display_title": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/add_global_descriptors.py",
>       "path": "pipeline/analysis/add_global_descriptors.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/add_global_descriptors.py",
>       "display_title": "pipeline/analysis/add_global_descriptors.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_morfeus_descriptors.py",
>       "path": "cross_benzoin/analysis/add_morfeus_descriptors.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/add_morfeus_descriptors.py",
>       "display_title": "cross_benzoin/analysis/add_morfeus_descriptors.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analyze_benzoin_deep.py",
>       "path": "pipeline/analyze_benzoin_deep.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analyze_benzoin_deep.py",
>       "display_title": "pipeline/analyze_benzoin_deep.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_backfill_mwf_cb_array.sh",
>       "path": "cross_benzoin/slurm/submit_backfill_mwf_cb_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_backfill_mwf_cb_array.sh",
>       "display_title": "cross_benzoin/slurm/submit_backfill_mwf_cb_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/concat_mordred_chunks.py",
>       "path": "cross_benzoin/analysis/concat_mordred_chunks.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/concat_mordred_chunks.py",
>       "display_title": "cross_benzoin/analysis/concat_mordred_chunks.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_bvalid_product_descriptors_20260623.md",
>       "path": "pipeline/docs/REPORT_bvalid_product_descriptors_20260623.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/REPORT_bvalid_product_descriptors_20260623.md",
>       "display_title": "pipeline/docs/REPORT_bvalid_product_descriptors_20260623.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_cb_train.sh",
>       "path": "pipeline/slurm/submit_cb_train.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_cb_train.sh",
>       "display_title": "pipeline/slurm/submit_cb_train.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "path": "README.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "display_title": "README.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/clean_and_diagnose_homo_v6.py",
>       "path": "cross_benzoin/analysis/clean_and_diagnose_homo_v6.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/analysis/clean_and_diagnose_homo_v6.py",
>       "display_title": "cross_benzoin/analysis/clean_and_diagnose_homo_v6.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "path": "pipeline/build_cb_training_table.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "display_title": "pipeline/build_cb_training_table.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/viz_gxtb_products_full.py",
>       "path": "pipeline/analysis/viz_gxtb_products_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/viz_gxtb_products_full.py",
>       "display_title": "pipeline/analysis/viz_gxtb_products_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_baseline72.py",
>       "path": "pipeline/analysis/shap_baseline72.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/shap_baseline72.py",
>       "display_title": "pipeline/analysis/shap_baseline72.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "path": "pipeline/analysis/train_bde_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_bde_surrogate.py",
>       "display_title": "pipeline/analysis/train_bde_surrogate.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/add_global_both.py",
>       "path": "pipeline/analysis/add_global_both.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/add_global_both.py",
>       "display_title": "pipeline/analysis/add_global_both.py"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "acceptor_id donor_id",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 25
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "path": "cross_benzoin/ARCHITECTURE.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "display_title": "cross_benzoin/ARCHITECTURE.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/RUNBOOK_v6_products.md",
>       "path": "pipeline/RUNBOOK_v6_products.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/RUNBOOK_v6_products.md",
>       "display_title": "pipeline/RUNBOOK_v6_products.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/predict_library.py",
>       "path": "pipeline/predict_library.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/predict_library.py",
>       "display_title": "pipeline/predict_library.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "path": "cross_benzoin/cb_featurize.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "display_title": "cross_benzoin/cb_featurize.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/bvalid_product_eval.py",
>       "path": "pipeline/bvalid_product_eval.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/bvalid_product_eval.py",
>       "display_title": "pipeline/bvalid_product_eval.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "path": "pipeline/build_cb_training_table.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "display_title": "pipeline/build_cb_training_table.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "path": "cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "display_title": "cross_benzoin/slurm/submit_cb_featurize_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/featurize_product.py",
>       "path": "pipeline/compute/featurize_product.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/featurize_product.py",
>       "display_title": "pipeline/compute/featurize_product.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/PLAN_full_library_cb_20260623.md",
>       "path": "pipeline/docs/PLAN_full_library_cb_20260623.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/PLAN_full_library_cb_20260623.md",
>       "display_title": "pipeline/docs/PLAN_full_library_cb_20260623.md"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "cross_benzoin donor acceptor descriptors",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 25
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "path": "cross_benzoin/ARCHITECTURE.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/ARCHITECTURE.md",
>       "display_title": "cross_benzoin/ARCHITECTURE.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "path": "cross_benzoin/cb_featurize.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/cb_featurize.py",
>       "display_title": "cross_benzoin/cb_featurize.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "path": "cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/cross_benzoin/slurm/submit_cb_featurize_array.sh",
>       "display_title": "cross_benzoin/slurm/submit_cb_featurize_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/RUNBOOK_v6_products.md",
>       "path": "pipeline/RUNBOOK_v6_products.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/RUNBOOK_v6_products.md",
>       "display_title": "pipeline/RUNBOOK_v6_products.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "path": "pipeline/build_cb_training_table.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/build_cb_training_table.py",
>       "display_title": "pipeline/build_cb_training_table.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/WORKFLOW_homo_v6_simulation.md",
>       "path": "pipeline/docs/WORKFLOW_homo_v6_simulation.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/docs/WORKFLOW_homo_v6_simulation.md",
>       "display_title": "pipeline/docs/WORKFLOW_homo_v6_simulation.md"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.search
>
> ```json
> {
>   "query": "C-C BDE cross",
>   "repository_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "topn": 25
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "results": [
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bde_array.sh",
>       "path": "pipeline/slurm/submit_bde_array.sh",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/slurm/submit_bde_array.sh",
>       "display_title": "pipeline/slurm/submit_bde_array.sh"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde.py",
>       "path": "pipeline/compute/calc_bde.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/compute/calc_bde.py",
>       "display_title": "pipeline/compute/calc_bde.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bde.py",
>       "path": "pipeline/analysis/finalize_correction_bde.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bde.py",
>       "display_title": "pipeline/analysis/finalize_correction_bde.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "path": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/train_prod_bdfe_smiles_surrogate.py",
>       "display_title": "pipeline/analysis/train_prod_bdfe_smiles_surrogate.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "path": "README.md",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/README.md",
>       "display_title": "README.md"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_product_bdfe.py",
>       "path": "pipeline/analysis/analyze_product_bdfe.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/analyze_product_bdfe.py",
>       "display_title": "pipeline/analysis/analyze_product_bdfe.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "path": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py",
>       "display_title": "pipeline/analysis/finalize_correction_bdfe_gxtb_full.py"
>     },
>     {
>       "url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "path": "pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/d926d1820a3e8d42eccc429cd84c68e26bb7181d/pipeline/analysis/gnn_dual_qm_champion275.py",
>       "display_title": "pipeline/analysis/gnn_dual_qm_champion275.py"
>     }
>   ]
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "cross_benzoin/cb_featurize.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python3\n\"\"\"\ncross_benzoin unified featurizer  (see ARCHITECTURE.md)\n=======================================================\nOne entry point for benzoin PRODUCT featurization (homo = diagonal special case).\nAldehyde and product share the SAME funnel_v3 method + descriptor backends, and\nALL geometries/energies/descriptors are saved and cross-linked by stable IDs.\n\nReuses (does not re-implement) the validated backends in pipeline/compute:\n  ald_descriptors_qm (xTB/morfeus/Multiwfn), thermo_orca (ohess G),\n  conf_funnel_v3 (conformer search), featurize_product (benzoin-core logic).\n\nInputs (one of):\n  --pairs PAIRS.csv      columns: donor_id,acceptor_id,donor_smiles,acceptor_smiles\n  --homo-from LIB.csv    aldehyde library (index,SMILES[,xtb_optimized]) -> homo pairs\n\nOutput: a RUN DIRECTORY with fixed names:\n  <out>/aldehydes.csv  <out>/products.csv  <out>/xyz_ald/  <out>/xyz_prod/\n\nExample:\n  python cb_featurize.py --homo-from data/library/aldehydes_clean_v6.csv \\\n      --out data/cross_benzoin/homo_v6 --emit-aldehydes --multiwfn \\\n      --multiwfn-bin /home/schen3/mutiwfn/Multiwfn_noGUI --workers 12 --n-confs 10\n\"\"\"\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport logging\nimport os\nimport re\nimport shutil\nimport subprocess\nimport sys\nfrom concurrent.futures import ProcessPoolExecutor, as_completed\nfrom pathlib import Path\n\n# make the validated backends importable\n_COMPUTE = Path(__file__).resolve().parents[1] / \"pipeline\" / \"compute\"\nsys.path.insert(0, str(_COMPUTE))\n# conf_funnel_v3 MUST be imported BEFORE thermo_orca / ald_descriptors_qm: thermo_orca and\n# conf_funnel_v2 are mutually circular, and loading conf_funnel_v3 first fully initialises\n# thermo_orca via that path. The old order (ald/thermo first) crashed with\n# \"cannot import name '_mol_rotbonds' from partially initialized module 'thermo_orca'\".\nimport conf_funnel_v3                   # noqa: E402\nimport ald_descriptors_qm as A          # noqa: E402\nimport thermo_orca as Th                # noqa: E402\nimport featurize_product as FP          # noqa: E402  (benzoin-core + product calc_* reuse)\n\nfrom rdkit import Chem                  # noqa: E402\n\nlog = logging.getLogger(\"cb_featurize\")\nHARTREE_TO_KCAL = 627.509474\n\n# ── g-xTB baseline, FUSED into this pass ─────────────────────────────────────\n# The production Δ-model baseline is g-xTB. Computing it as a separate job would redo the\n# whole funnel_v3 + GFN2-ohess geometry (the expensive part) a SECOND time. Instead we do\n# one g-xTB COSMO(DMSO) SP on the GFN2-ohess geometry we already have → G_gxtb in the same\n# pass (≈+10-20% vs ~2× cost). Mirrors pipeline/compute/gxtb_baseline.py. Needs GXTB_BIN\n# (+ GXTB_SOLV) in the env, exactly as submit_gxtb_baseline.sh sets them.\nXTB_GXTB = os.environ.get(\n    \"GXTB_BIN\", \"/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb\")\nGXTB_SOLV = os.environ.get(\"GXTB_SOLV\", \"cosmo dmso\").split()  # ALPB/GBSA fatal; COSMO ~ CPCM\n_GXTB_E = re.compile(r\"::\\s*total energy\\s+(-?\\d+\\.\\d+)\\s+Eh\")\n\n\ndef _gxtb_sp(geom: Path, wd: Path, charge: int = 0, timeout: int = 900) -> float | None:\n    \"\"\"g-xTB COSMO(DMSO) single point on `geom` → total energy E_gxtb (Eh), None on failure.\"\"\"\n    wd.mkdir(parents=True, exist_ok=True)\n    (wd / \"g.xyz\").write_text(Path(geom).read_text())\n    cmd = [XTB_GXTB, \"g.xyz\", \"--gxtb\", \"--sp\", \"--chrg\", str(charge)]\n    if GXTB_SOLV:\n        cmd += [\"--\" + GXTB_SOLV[0], *GXTB_SOLV[1:]]\n    try:\n        r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout)\n    except subprocess.TimeoutExpired:\n        return None\n    (wd / \"gxtb_sp.log\").write_text(r.stdout + r.stderr)\n    m = _GXTB_E.findall(r.stdout + r.stderr)\n    return float(m[-1]) if m else None\n\n\ndef _g_gxtb(ohess_stdout: str, ohess_dir: Path, smiles: str, wd: Path) -> float | None:\n    \"\"\"G_gxtb (Eh) = E_gxtb + (G_gfn2 − E_el_gfn2): g-xTB electronic energy on the GFN2-ohess\n    geometry, reusing the GFN2 RRHO thermal correction. None if any piece is missing.\"\"\"\n    G = Th.parse_xtb_G(ohess_stdout)\n    E_el = Th._parse_xtb_energy(ohess_stdout)\n    geom = ohess_dir / \"xtbopt.xyz\"\n    if G is None or E_el is None or not geom.exists():\n        return None\n    try:\n        m = Chem.MolFromSmiles(smiles)\n        chg = Chem.GetFormalCharge(m) if m is not None else 0\n    except Exception:\n        chg = 0\n    E_gxtb = _gxtb_sp(geom, wd / \"gxtb\", charge=chg)\n    return E_gxtb + (G - E_el) if E_gxtb is not None else None\n\n# ── Schema (single source of truth) ─────────────────────────────────────────\n_ALD_DESC = [c for c in A._ALL_FIELDS\n             if c not in (\"index\", \"SMILES\", \"PubChem_CID\",\n                          \"xtb_optimized\", \"error\", \"xyz_file\")]\nALD_FIELDS = [\"id\", \"smiles\", \"xtb_optimized\", \"error\", \"xyz_file\", \"G_xtb\", \"G_gxtb\"] + _ALD_DESC\n\n_PROD_DESC = FP._XTB + FP._MORF + FP._MWF\nPROD_FIELDS = ([\"id\", \"donor_id\", \"acceptor_id\", \"donor_smiles\", \"acceptor_smiles\",\n                \"smiles\", \"reaction_type\", \"is_homo\", \"xtb_optimized\", \"error\", \"xyz_file\"]\n               + _PROD_DESC + [\"G_donor\", \"G_acceptor\", \"G_xtb\", \"dG_xtb_kcal\",\n                               \"G_donor_gxtb\", \"G_acceptor_gxtb\", \"G_gxtb\", \"dG_gxtb_kcal\"])\n\n\ndef pair_id(did: str, aid: str) -> str:\n    # homo (donor==acceptor) → single id, not the redundant \"<id>__<id>\"\n    return did if did == aid else f\"{did}__{aid}\"\n\n\ndef _safe(s: str) -> str:\n    return \"\".join(c if c.isalnum() or c in \"-._\" else \"_\" for c in str(s))[:60]\n\n\ndef _rank(name: str):\n    return conf_funnel_v3.rank_conformers_funnel_v3 if name == \"funnel_v3\" else Th._rank_conformers\n\n\n# ── Aldehyde: funnel_v3 geometry (saved) + descriptors + G ───────────────────\ndef featurize_aldehyde(ald_id, smi, *, xyz_dir, work_dir, xtb_bin, mwf_bin, do_multiwfn,\n                       solvent, n_confs, T, P, cores, jobs, timeout, conformer):\n    row = {f: None for f in ALD_FIELDS}\n    row.update({\"id\": str(ald_id), \"smiles\": smi, \"xtb_optimized\": False, \"error\": \"\"})\n    wd = work_dir / f\"ald_{_safe(ald_id)}\"\n    # Free this aldehyde's scratch (conf/desc/mwf/ohess — hundreds of tiny xTB+Multiwfn\n    # files) before returning; row + saved xyz are all that's kept. work_dir is on the\n    # shared gpfs:scratch1/nodespecific tree with ONE per-user inode quota across all\n    # nodes, so without per-molecule cleanup a full 220k chunk array (esp. with Multiwfn,\n    # and at high %throttle) exhausts it mid-run with Errno 122. The chunk-end rmtree is\n    # too late — all molecules' scratch coexists until then.\n    try:\n        ranked = _rank(conformer)(smi, wd / \"conf\", xtb_bin, n_confs, \"ald\",\n                                  solvent=solvent, cores=cores, workers=jobs)\n        if not ranked:\n            row[\"error\"] = \"ald_embed_failed\"\n            return row, None\n        best = ranked[0][0]\n        row[\"xtb_optimized\"] = True\n        xp = xyz_dir / f\"ald_{_safe(ald_id)}.xyz\"\n        xp.write_text(best, encoding=\"utf-8\")\n        row[\"xyz_file\"] = str(xp)\n        sym, coord = A.parse_xyz(best)\n        desc = {}\n        desc.update(A.calc_xtb(best, sym, coord, xtb_bin, wd / \"desc\"))\n        desc.update(A.calc_morfeus(sym, coord))\n        if do_multiwfn and mwf_bin:\n            desc.update(A.calc_multiwfn(best, sym, coord, xtb_bin, mwf_bin,\n                                        wd / \"mwf\", stem=f\"ald_{_safe(ald_id)}\"))\n        for k, v in desc.items():\n            if k in row:\n                row[k] = v\n        sa, _ = Th.run_ohess(best, wd / \"ohess\", xtb_bin, T, P, solvent=solvent,\n                             cores=cores, timeout=timeout)\n        G = Th.parse_xtb_G(sa)\n        row[\"G_xtb\"] = G\n        G_gxtb = _g_gxtb(sa, wd / \"ohess\", smi, wd)   # g-xTB SP on the same ohess geom\n        row[\"G_gxtb\"] = G_gxtb\n        return row, (G, G_gxtb)\n    finally:\n        shutil.rmtree(wd, ignore_errors=True)\n\n\n# ── Product: build, funnel_v3 geometry (saved) + descriptors + ΔG ────────────\ndef featurize_pair(rec, *, g_cache, xyz_dir, work_dir, xtb_bin, mwf_bin, do_multiwfn,\n                   solvent, n_confs, T, P, cores, jobs, timeout, conformer):\n    did = str(rec.get(\"donor_id\") or rec.get(\"index\") or \"d\")\n    aid = str(rec.get(\"acceptor_id\") or did)\n    donor = (rec.get(\"donor_smiles\") or \"\").strip()\n    acc = (rec.get(\"acceptor_smiles\") or \"\").strip()\n    pid = pair_id(did, aid)\n    row = {f: None for f in PROD_FIELDS}\n    row.update({\"id\": pid, \"donor_id\": did, \"acceptor_id\": aid,\n                \"donor_smiles\": donor, \"acceptor_smiles\": acc,\n                \"xtb_optimized\": False, \"error\": \"\"})\n    if not donor or not acc:\n        row[\"error\"] = \"missing_smiles\"\n        return row\n    is_homo = Chem.CanonSmiles(donor) == Chem.CanonSmiles(acc)\n    row[\"is_homo\"] = is_homo\n    row[\"reaction_type\"] = FP.reaction_type(FP.classify(donor), FP.classify(acc), is_homo)\n    prod = FP.build_product(donor, acc)\n    if not prod:\n        row[\"error\"] = \"product_build_failed\"\n        return row\n    row[\"smiles\"] = prod\n    wd = work_dir / f\"prod_{_safe(pid)}\"\n    # Per-product scratch cleanup — see featurize_aldehyde: bounds the live inode\n    # footprint to ~workers concurrent molecules instead of a whole chunk, which is\n    # what keeps a 220k funnel_v3 array under the shared nodespecific inode quota.\n    try:\n        ranked = _rank(conformer)(prod, wd / \"conf\", xtb_bin, n_confs, \"prod\",\n                                  solvent=solvent, cores=cores, workers=jobs)\n        if not ranked:\n            row[\"error\"] = \"prod_embed_failed\"\n            return row\n        best = ranked[0][0]\n        row[\"xtb_optimized\"] = True\n        _stem = _safe(did) if did == aid else f\"{_safe(did)}__{_safe(aid)}\"\n        xp = xyz_dir / f\"prod_{_stem}.xyz\"\n        xp.write_text(best, encoding=\"utf-8\")\n        row[\"xyz_file\"] = str(xp)\n        sym, coord = A.parse_xyz(best)\n        core = FP.find_benzoin_core(sym, coord)\n        if core is None:\n            row[\"error\"] = \"core_not_found\"\n            return row\n        descs = [FP.calc_xtb_product(best, sym, coord, core, xtb_bin, wd / \"desc\"),\n                 FP.calc_morfeus_product(sym, coord, core)]\n        if do_multiwfn and mwf_bin:\n            descs.append(FP.calc_multiwfn_product(best, sym, coord, core, xtb_bin, mwf_bin,\n                                                  wd / \"mwf\", pid))\n        for d in descs:\n            for k, v in d.items():\n                if k in row:\n                    row[k] = v\n        sp, _ = Th.run_ohess(best, wd / \"ohess\", xtb_bin, T, P, solvent=solvent,\n                             cores=cores, timeout=timeout)\n        Gp = Th.parse_xtb_G(sp)\n        Gp_g = _g_gxtb(sp, wd / \"ohess\", prod, wd)            # product g-xTB G\n        Gd, Gd_g = g_cache.get(Chem.CanonSmiles(donor)) or (None, None)\n        Ga, Ga_g = g_cache.get(Chem.CanonSmiles(acc)) or (None, None)\n        row[\"G_donor\"], row[\"G_acceptor\"], row[\"G_xtb\"] = Gd, Ga, Gp\n        row[\"G_donor_gxtb\"], row[\"G_acceptor_gxtb\"], row[\"G_gxtb\"] = Gd_g, Ga_g, Gp_g\n\n        def _flag(extra: str) -> None:\n            row[\"error\"] = (row[\"error\"] + \";\" if row[\"error\"] else \"\") + extra\n\n        if None not in (Gp, Gd, Ga):\n            row[\"dG_xtb_kcal\"] = round((Gp - Gd - Ga) * HARTREE_TO_KCAL, 4)\n        else:\n            _flag(\"dG_failed\")\n        if None not in (Gp_g, Gd_g, Ga_g):\n            row[\"dG_gxtb_kcal\"] = round((Gp_g - Gd_g - Ga_g) * HARTREE_TO_KCAL, 4)\n        else:\n            # Distinguish WHERE the g-xTB chain broke: the product SP itself vs a\n            # missing/failed g-xTB value on a cached donor or acceptor (the GFN2 side\n            # can still be fine, so this must not clobber a successful dG_xtb_kcal above).\n            if Gp_g is None:\n                _flag(\"gxtb_sp_failed\")\n            if Gd_g is None or Ga_g is None:\n                _flag(\"gxtb_dG_failed_reactant\")\n        return row\n    finally:\n        shutil.rmtree(wd, ignore_errors=True)\n\n\n# ── Inputs ───────────────────────────────────────────────────────────────────\ndef load_pairs(args) -> list[dict]:\n    if args.pairs:\n        with open(args.pairs, encoding=\"utf-8\") as fh:\n            return list(csv.DictReader(fh))\n    rows = []\n    with open(args.homo_from, encoding=\"utf-8\") as fh:\n        for r in csv.DictReader(fh):\n            smi = (r.get(\"SMILES\") or r.get(\"smiles\") or \"\").strip()\n            if not smi:\n                continue\n            if \"xtb_optimized\" in r and str(r[\"xtb_optimized\"]).strip() not in (\"\", \"True\", \"true\", \"1\"):\n                continue\n            i = str(r.get(\"index\") or r.get(\"id\") or len(rows))\n            rows.append({\"donor_id\": i, \"acceptor_id\": i,\n                         \"donor_smiles\": smi, \"acceptor_smiles\": smi})\n    return rows\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser(description=__doc__,\n                                 formatter_class=argparse.RawDescriptionHelpFormatter)\n    src = ap.add_mutually_exclusive_group(required=True)\n    src.add_argument(\"--pairs\", help=\"pairs CSV: donor_id,acceptor_id,donor_smiles,acceptor_smiles\")\n    src.add_argument(\"--homo-from\", help=\"aldehyde library CSV (index,SMILES) -> homo pairs\")\n    ap.add_argument(\"--out\", required=True, help=\"run output directory\")\n    ap.add_argument(\"--emit-aldehydes\", action=\"store_true\",\n                    help=\"also featurize+save each unique aldehyde (funnel_v3)\")\n    ap.add_argument(\"--xtb-bin\", default=shutil.which(\"xtb\") or \"/home/schen3/xtb/bin/xtb\")\n    ap.add_argument(\"--multiwfn\", action=\"store_true\")\n    ap.add_argument(\"--multiwfn-bin\", default=\"/home/schen3/mutiwfn/Multiwfn_noGUI\")\n    ap.add_argument(\"--conformer\", choices=[\"funnel_v3\", \"rank\"], default=\"funnel_v3\")\n    ap.add_argument(\"--solvent\", default=\"dmso\")\n    ap.add_argument(\"--n-confs\", type=int, default=10)\n    ap.add_argument(\"--xtb-cores\", type=int, default=2)\n    ap.add_argument(\"--parallel-jobs\", type=int, default=1)\n    ap.add_argument(\"--ohess-timeout\", type=int, default=900)\n    ap.add_argument(\"--T\", type=float, default=298.15)\n    ap.add_argument(\"--P\", type=float, default=1.0)\n    ap.add_argument(\"--workers\", type=int, default=1)\n    ap.add_argument(\"--max\", type=int, default=0)\n    args = ap.parse_args()\n\n    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(message)s\", datefmt=\"%H:%M:%S\")\n    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin\n    solvent = \"\" if args.solvent.lower() == \"none\" else args.solvent\n    do_multiwfn = args.multiwfn and bool(args.multiwfn_bin)\n\n    pairs = load_pairs(args)\n    if args.max:\n        pairs = pairs[: args.max]\n\n    out = Path(args.out)\n    xyz_ald, xyz_prod = out / \"xyz_ald\", out / \"xyz_prod\"\n    work_dir = Path(os.environ.get(\"TMPDIR\", \"/tmp\")) / f\"cb_featurize_{os.getpid()}\"\n    for d in (out, xyz_prod, work_dir):\n        d.mkdir(parents=True, exist_ok=True)\n    if args.emit_aldehydes:\n        xyz_ald.mkdir(parents=True, exist_ok=True)\n    log.info(\"pairs=%d  emit_aldehydes=%s  multiwfn=%s  conformer=%s  out=%s\",\n             len(pairs), args.emit_aldehydes, do_multiwfn, args.conformer, out)\n\n    # unique aldehydes: CanonSmiles -> (ald_id, smiles)\n    uniq: dict[str, tuple[str, str]] = {}\n    for r in pairs:\n        for role in (\"donor\", \"acceptor\"):\n            smi = (r.get(f\"{role}_smiles\") or \"\").strip()\n            if not smi or not Chem.MolFromSmiles(smi):\n                continue\n            uniq.setdefault(Chem.CanonSmiles(smi), (str(r.get(f\"{role}_id\") or \"a\"), smi))\n\n    g_cache: dict[str, float | None] = {}\n\n    # ── aldehyde phase ────────────────────────────────────────────────────────\n    akw = dict(xyz_dir=xyz_ald, work_dir=work_dir, xtb_bin=xtb_bin, mwf_bin=args.multiwfn_bin,\n               do_multiwfn=do_multiwfn, solvent=solvent, n_confs=args.n_confs, T=args.T, P=args.P,\n               cores=args.xtb_cores, jobs=args.parallel_jobs, timeout=args.ohess_timeout,\n               conformer=args.conformer)\n    items = list(uniq.items())  # [(canon, (id, smi)), ...]\n    log.info(\"unique aldehydes: %d\", len(items))\n    if args.emit_aldehydes:\n        with open(out / \"aldehydes.csv\", \"w\", newline=\"\", encoding=\"utf-8\") as afh:\n            aw = csv.DictWriter(afh, fieldnames=ALD_FIELDS, extrasaction=\"ignore\")\n            aw.writeheader()\n            with ProcessPoolExecutor(max_workers=args.workers) as ex:\n                futs = {ex.submit(featurize_aldehyde, aid, smi, **akw): canon\n                        for canon, (aid, smi) in items}\n                for n, fut in enumerate(as_completed(futs), 1):\n                    canon = futs[fut]\n                    try:\n                        prow, gpair = fut.result()      # gpair = (G_gfn2, G_gxtb)\n                    except Exception as exc:\n                        prow, gpair = {\"smiles\": canon, \"error\": f\"exception:{exc}\"}, None\n                    aw.writerow(prow); afh.flush(); g_cache[canon] = gpair\n                    if n % 25 == 0 or n == len(items):\n                        log.info(\"  ald %d/%d\", n, len(items))\n    else:  # only need G for ΔG (GFN2 only; g-xTB unavailable on this path → None)\n        with ProcessPoolExecutor(max_workers=args.workers) as ex:\n            futs = {ex.submit(FP.ald_free_energy, smi, work_dir / f\"aldG_{_safe(aid)}\", xtb_bin,\n                              solvent, args.n_confs, args.T, args.P, args.xtb_cores,\n                              args.parallel_jobs, args.ohess_timeout, args.conformer): canon\n                    for canon, (aid, smi) in items}\n            for n, fut in enumerate(as_completed(futs), 1):\n                try:\n                    res = fut.result()                  # ald_free_energy → (G, xyz)\n                    g_cache[futs[fut]] = (res[0] if isinstance(res, tuple) else res, None)\n                except Exception:\n                    g_cache[futs[fut]] = None\n\n    # ── product phase ─────────────────────────────────────────────────────────\n    pkw = dict(g_cache=g_cache, xyz_dir=xyz_prod, work_dir=work_dir, xtb_bin=xtb_bin,\n               mwf_bin=args.multiwfn_bin, do_multiwfn=do_multiwfn, solvent=solvent,\n               n_confs=args.n_confs, T=args.T, P=args.P, cores=args.xtb_cores,\n               jobs=args.parallel_jobs, timeout=args.ohess_timeout, conformer=args.conformer)\n    n_ok = n_dg = n_err = 0\n    with open(out / \"products.csv\", \"w\", newline=\"\", encoding=\"utf-8\") as pfh:\n        pw = csv.DictWriter(pfh, fieldnames=PROD_FIELDS, extrasaction=\"ignore\")\n        pw.writeheader()\n        with ProcessPoolExecutor(max_workers=args.workers) as ex:\n            futs = {ex.submit(featurize_pair, r, **pkw): i for i, r in enumerate(pairs)}\n            for fut in as_completed(futs):\n                try:\n                    row = fut.result()\n                except Exception as exc:\n                    row = {\"id\": str(futs[fut]), \"error\": f\"exception:{exc}\"}\n                pw.writerow(row); pfh.flush()\n                n_ok += 1; n_dg += row.get(\"dG_xtb_kcal\") is not None; n_err += bool(row.get(\"error\"))\n    shutil.rmtree(work_dir, ignore_errors=True)\n    log.info(\"done: %d products (%d with dG, %d errors) -> %s/products.csv\", n_ok, n_dg, n_err, out)\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n",
>   "encoding": "utf-8",
>   "sha": "c39079911b6e21f75142a24f7bdff4d2e9ba70bc",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/cross_benzoin/cb_featurize.py",
>   "display_title": "cb_featurize.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "cross_benzoin/ARCHITECTURE.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ````json
> {
>   "content": "# cross_benzoin/ — architecture & naming conventions\n\nClean, single-entry pipeline for NHC benzoin **product** featurization (homo =\ndiagonal special case of cross). Replaces the scattered\n`pipeline/compute/{featurize_screen,featurize_product,backfill_multiwfn,merge_multiwfn}.py`\nfor everything downstream of the aldehyde library. Reuses the validated QM\nbackends (does NOT re-implement them):\n\n- `pipeline/compute/ald_descriptors_qm.py` — xTB / morfeus / Multiwfn calculators\n- `pipeline/compute/thermo_orca.py`        — conformer ranking, xTB `--ohess` G\n- `pipeline/compute/conf_funnel_v3.py`     — funnel_v3 conformer search (topology guard)\n\n## Principle\n\nONE method, everything saved. Aldehyde and product are featurized with the SAME\nconformer method (funnel_v3) and identical descriptor backends, and all geometries\n+ energies + descriptors are persisted and cross-linked by a stable ID.\n\n## Stable IDs  (the linking key — never an enumerate index)\n\n- `ald_id`  : stable per-aldehyde id taken from the library column `index`\n              (falls back to InChIKey). Used in aldehydes.csv and xyz filenames.\n- `pair_id` : `\"{donor_id}__{acceptor_id}\"` (ordered). HOMO (donor==acceptor)\n              collapses to the single `donor_id` (no redundant `<id>__<id>`).\n              Used in products.csv and product xyz filenames.\n\n## File / directory layout\n\n```\nbenzoin-dg/cross_benzoin/                # code (this package)\n  ARCHITECTURE.md\n  cb_schema.py          # column lists + naming helpers (single source of truth)\n  cb_featurize.py       # the one entry point (aldehyde and/or product)\n  slurm/submit_cb_featurize_array.sh\n  analysis/\n\nbenzoin-dg/data/cross_benzoin/<run>/     # outputs (e.g. <run> = homo_v6, cross_core_v1)\n  aldehydes.csv         # one row per aldehyde   (key: ald_id)\n  products.csv          # one row per pair        (key: pair_id, + donor_id, acceptor_id)\n  xyz_ald/ald_<ald_id>.xyz\n  xyz_prod/prod_<pair_id>.xyz   (homo: prod_<id>.xyz; cross: prod_<donor>__<acceptor>.xyz)\n  chunk_*/              # (array mode) per-task aldehydes.csv/products.csv/xyz_*; merged after\n```\n\n## File naming (unified)\n\n| artifact | name |\n|---|---|\n| aldehyde geometry | `xyz_ald/ald_<ald_id>.xyz` |\n| product geometry  | `xyz_prod/prod_<pair_id>.xyz   (homo: prod_<id>.xyz; cross: prod_<donor>__<acceptor>.xyz)` |\n| aldehyde table    | `aldehydes.csv` |\n| product table     | `products.csv` |\n| concatenated (array) | `aldehydes_all.csv`, `products_all.csv` |\n\n## Column conventions\n\nShared metadata (identical names in both tables):\n`id` (ald_id or pair_id), `smiles`, `xtb_optimized`, `error`, `xyz_file`, `G_xtb` (Eh).\n\nAldehyde descriptors — single carbonyl site, suffix `_CHO_C` / `_CHO_O`\n(same as the screen schema, so existing analysis still works).\n\nProduct descriptors — two derived sites + the H-bond, suffixes:\n`_ketC` `_ketO` (former donor carbonyl → ketone), `_carbC` `_hydO` `_hydH`\n(former acceptor carbonyl → carbinol), bonds `_CO_ket` `_CC_new` `_CO_carb`,\nH-bond `hb_dist` `hb_angle` `qtaim_rho_HB`. ΔG column `dG_xtb_kcal`.\n\n## Data flow\n\n```\naldehyde library CSV (index, SMILES, ...)\n        │   pairs CSV: donor_id, acceptor_id, donor_smiles, acceptor_smiles\n        ▼\ncb_featurize.py  (funnel_v3, --multiwfn, --emit-aldehydes)\n        ├─ per unique aldehyde → aldehydes.csv row + xyz_ald/ald_<id>.xyz\n        └─ per pair            → products.csv row  + xyz_prod/prod_<did>__<aid>.xyz\n                                 ΔG = G(prod) − G(donor) − G(acceptor)  (G reused from aldehydes)\n```\n\n## Migration / status\n\n- v6 homo campaign and all future cross runs use THIS package.\n- Old scripts kept for provenance; `backfill_multiwfn`/`merge_multiwfn` are\n  deprecated (plain-geometry, method-inconsistent).\n- See [[cross-benzoin-pipeline-handoff]], [[multiwfn-env-and-screen-gap]].\n",
>   "encoding": "utf-8",
>   "sha": "97949fcdf3490bca2dcb55af0b13169b876b8e23",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/cross_benzoin/ARCHITECTURE.md",
>   "display_title": "ARCHITECTURE.md"
> }
> ````
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/docs/PLAN_full_library_cb_20260623.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# 全库 cb featurization — 点火前评估 (2026-06-23)\n\n## 1. homo_pairs.csv —— 就绪 ✓\n`data/cross_benzoin/homo_v6/homo_pairs.csv`：**220,724 行**，列 `donor_id,acceptor_id,donor_smiles,acceptor_smiles`，\n正是 `submit_cb_featurize_array.sh` 期望的格式。来源 = screen_v6 中 `xtb_optimized==True` 的分子\n（merged CSV `data/raw/screen_v6/analysis/screen_v6_features_mwf_all.csv` 保留，chunk 目录已在配额清理时删除，不影响）。\n→ 输入侧无需再做。\n\n## 2. \"screen 的 funnel_v3 几何\" —— 已澄清，无需重跑 screen\n关键认识：**cb_featurize 从 SMILES 重新做 funnel_v3 几何，不复用 screen 的 legacy-ranker 几何。**\n- screen（legacy `_rank_conformers`）只是 **T2 triage / SMILES 来源**；homo_pairs 只带 SMILES+id，无几何。\n- cb_featurize 拿这些 SMILES **重新跑 funnel_v3**（ETKDG→GFN-FF→GFN2 opt+拓扑守卫）→ 几何与训练表（funnel_v3）一致。\n- 所以记忆里说的\"screen 需用 funnel_v3 重跑\"，真实含义是 **featurization 要用 funnel_v3 重做** = 这次全库 cb 本身。\n  **不需要**单独把 triage screen 重跑一遍。screen 的 legacy 几何只服务于已完成的 dG_xtb 粗筛，会被 cb 的 funnel_v3 取代。\n\n## 3. ⚠️ 最大优化点 —— g-xTB SP 必须并入 cb，否则几何算两遍 (~省 7 万核时)\n- 生产模型基线是 **g-xTB**（已 ship），所以 220k screen 需要每个分子的 **g-xTB ΔG**。\n- 但 **cb_featurize 现在只算 GFN2**（`dG_xtb_kcal`），不含 g-xTB（已确认 grep 无 gxtb）。\n- 而 `gxtb_baseline.py::_species()` **会重新做 funnel_v3 + GFN2 ohess + g-xTB SP** —— 与 cb **重复算同一套昂贵几何**。\n- 现成本（B 实测 **0.384 核时/对**）：\n  | 方案 | 核时 |\n  |---|---|\n  | cb_featurize 单独 (220,724 对) | **~84,800** |\n  | + gxtb_baseline 独立(再做一遍几何) | **再 +~85,000** → 合计 ~170,000 |\n  | **融合：cb 内几何上加一次 g-xTB SP** | **~95,000–100,000**（g-xTB SP 仅 +10–20%） |\n- **结论：把 g-xTB SP 加进 cb_featurize 的 featurize_aldehyde/featurize_pair**（ohess 几何上一次\n  `xtb --gxtb --sp --cosmo dmso`，G_gxtb = E_gxtb + (G_gfn2−E_gfn2)，输出 `dG_gxtb_kcal` 列）。\n  代码现成：`gxtb_baseline.py` 的 `_gxtb_sp()` + `_species()` 模式可直接搬。**省 ~7 万核时，且产物只算一遍几何。**\n\n## 4. 耗时 (融合方案, %128)\n- 2,207 chunks (CHUNK=100)，每 chunk ~中位 1.6h（B 50 对中位 48min × 2，含 g-xTB +~15%），重 chunk 可达 ~2.5h。\n- %128 并发：2207/128 ≈ 17.3 波 × ~1.6h ≈ **~28h 纯计算墙钟**，加排队 → **~1.5–2.5 天**。\n- 单 chunk 远在 12h walltime 内（安全）。\n\n## 5. 风险\n| 风险 | 评级 | 缓解 |\n|---|---|---|\n| **核时预算 ~10 万核时**（融合）/ ~17 万（不融合） | 高 | 先融合 g-xTB；先 pilot 验证再全开 |\n| **straggler 浪费**：24 核任务等单个 60 原子分子的 ohess(O(N³)) | 中 | 按 heavy-atom 排序分桶 / size-cap 构象数 |\n| **inode @%128**：峰值 ~128×12×单分子 scratch ≈ 46 万瞬态 inode | 中 | 已有 per-mol rmtree + MULTIWFN=0 + 节点自愈 + cron；点火前用节点 `mmlsquota` 确认配额余量 |\n| **大分子 OOM**：mem=48G/24核=2G/核，60 原子 ohess 可能超 → SIGKILL → 孤儿 | 中 | 调高 mem 或重分子降 workers；cron 兜底孤儿 |\n| **产物 SMILES 错误**（[[benzoin-generator-formyl-bug]]）：SMARTS 建产物，220k 下小错率=数千个错产物 | 中 | 点火前跑 `check_smiles.py`（[[smiles-check-workflow-gate]]，纯 RDKit，便宜） |\n| homo_pairs 220,724 vs 库 220,859（少 135） | 低 | = xtb_optimized 子集，正常 |\n\n## 6. 由 B 模拟得到的可改进点（按收益排序）\n1. **融合 g-xTB SP 进 cb**（#3）—— 省 ~7 万核时，单几何出 GFN2+g-xTB 双基线。**最高优先**。\n2. **先 pilot 后全开**：先交 3–5 个 chunk，验证 g-xTB 列正确 + 计时 + 0 quota + SMILES，再放 2207。\n   绝不盲开 ~10 万核时。\n3. **size-aware 分桶 / size-cap**：按 heavy-atom 数排序，大分子单独小 CHUNK 或限构象数 → 削平墙钟、减核浪费。\n4. **预检 SMILES**：对整张 homo_pairs 跑 check_smiles（反应物+产物），剔除/修正错产物后再算。\n5. **重 chunk 的 mem/worker 调参**，避免 ohess OOM。\n6. （次要）醛阶段加心跳日志（现仅 25/50、50/50 两行，看着像静默）。\n\n## 推荐执行顺序\n1. 把 g-xTB SP 融进 cb_featurize（+ `dG_gxtb_kcal` 列）。\n2. `check_smiles.py` 预检 homo_pairs；节点上 `mmlsquota` 确认 inode 余量。\n3. **Pilot**：`--array=0-4%5` 跑 5 chunk，核对输出列(GFN2+g-xTB dG)、计时、0 quota、SMILES。\n4. 全开 `--array=0-2206%128`（MULTIWFN=0, EMIT_ALD=1, CONFORMER=funnel_v3）。\n5. 期间保持 orphan_cron 存活；跑完 concat → 接 g-xTB 基线模型 predict 全库。\n",
>   "encoding": "utf-8",
>   "sha": "1f6bbb929017349914a77f53949b799210b2641e",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/docs/PLAN_full_library_cb_20260623.md",
>   "display_title": "PLAN_full_library_cb_20260623.md"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "pipeline/build_cb_training_table.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "#!/usr/bin/env python\n\"\"\"Build the cross-benzoin Δ-learning training table (product-descriptor era).\n\nJoins the existing DFT labels (dG_orca_kcal) onto the NEW cross-benzoin featurization\n(aldehyde QM + product QM descriptors, no RDKit 2D), keyed on canonical aldehyde SMILES.\nThis is the training table for the current modeling direction (user 2026-06-24: no RDKit,\nproduct descriptors now computed). See [[cb-postfeaturize-autochain]] / [[descriptor-slim-v4]].\n\nConstruction choices (documented for review):\n  * Feature set = aldehyde QM descriptors (prefix `ald_`) + product QM descriptors\n    (prefix `prod_`) + `dG_xtb_kcal`. No RDKit 2D block. Multiwfn (adch_/qtaim_) cols\n    are all-empty at full scale and are dropped.\n  * Δ-target baseline `dG_xtb_kcal` is taken from the CROSS-BENZOIN run (not the label\n    run) so train and the 220k-prediction share the same baseline geometry/method. The\n    label vs cross-benzoin dG_xtb agree at bias 0.14 / r 0.93 (conformer-search scatter,\n    MAE ~1 kcal); delta_core's MAD QC trims the rare large geometry-mismatch outliers.\n  * Output keeps `SMILES` (canonical) so delta_core's scope/reactive filters work.\n\nOutput: data/featurize_cb_homo_train.parquet  (non-destructive; new file).\n\"\"\"\nfrom __future__ import annotations\nimport argparse\nfrom pathlib import Path\nimport pandas as pd\nfrom rdkit import Chem\nfrom rdkit import RDLogger; RDLogger.DisableLog(\"rdApp.*\")\n\nREPO = Path(\"/scratch-shared/schen3/benzoin-dg\")\nOUT = REPO / \"data/cross_benzoin/homo_v6\"\n# --baseline gfn2 -> dG_xtb_kcal ; gxtb -> dG_gxtb_kcal (relabelled to dG_xtb_kcal so the\n# existing delta_core pipeline treats it as the XTB_DG baseline unchanged).\nBASELINE_COL = {\"gfn2\": \"dG_xtb_kcal\", \"gxtb\": \"dG_gxtb_kcal\"}\n\nALD_DROP = {\"id\", \"smiles\", \"xtb_optimized\", \"error\", \"xyz_file\", \"G_xtb\", \"G_gxtb\"}\nPROD_DROP = {\"id\", \"donor_id\", \"acceptor_id\", \"donor_smiles\", \"acceptor_smiles\", \"smiles\",\n             \"reaction_type\", \"is_homo\", \"xtb_optimized\", \"error\", \"xyz_file\",\n             \"G_donor\", \"G_acceptor\", \"G_xtb\", \"G_donor_gxtb\", \"G_acceptor_gxtb\", \"G_gxtb\",\n             \"dG_xtb_kcal\", \"dG_gxtb_kcal\"}\n\n\ndef canon(s):\n    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None\n\n\ndef _numeric_feats(df, drop):\n    keep = []\n    for c in df.columns:\n        if c in drop:\n            continue\n        if c.startswith((\"adch_\", \"qtaim_\")):   # empty by design at full scale\n            continue\n        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():\n            keep.append(c)\n    return keep\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\"--baseline\", choices=[\"gfn2\", \"gxtb\"], default=\"gxtb\")\n    # authoritative r2SCAN-3c labels (user 2026-06-24): assembled from\n    # data/raw/featurize_funnelv3_relabel via data/labels_r2scan_relabel.parquet\n    ap.add_argument(\"--labels\", default=str(REPO / \"data/labels_r2scan_relabel.parquet\"))\n    args = ap.parse_args()\n    base_col = BASELINE_COL[args.baseline]\n    DEST = REPO / (f\"data/featurize_cb_homo_train_{args.baseline}.parquet\")\n    print(f\"baseline={args.baseline} (col {base_col}) labels={args.labels} -> {DEST.name}\")\n\n    lab = pd.read_parquet(args.labels)\n    lab = lab[lab[\"dG_orca_kcal\"].notna()].copy()\n    if \"c\" not in lab.columns:\n        lab[\"c\"] = lab[\"SMILES\"].map(canon)\n    lab = lab.dropna(subset=[\"c\"]).drop_duplicates(\"c\")\n    print(f\"labels (r2SCAN-3c): {len(lab)}  [ALL categories, no aromatic filter]\")\n\n    ald = pd.read_csv(OUT / \"aldehydes_all.csv\", low_memory=False)\n    ald[\"c\"] = ald[\"smiles\"].map(canon)\n    ald = ald.dropna(subset=[\"c\"]).drop_duplicates(\"c\")\n    af = _numeric_feats(ald, ALD_DROP)\n    ald_feat = ald.set_index(\"c\")[af].add_prefix(\"ald_\")\n    print(f\"aldehyde QM features: {len(af)}\")\n\n    prod = pd.read_csv(OUT / \"products_all.csv\", low_memory=False)\n    prod = prod[prod[\"error\"].isna()].copy()\n    prod[\"c\"] = prod[\"donor_smiles\"].map(canon)\n    prod = prod.dropna(subset=[\"c\"]).drop_duplicates(\"c\")\n    pf = _numeric_feats(prod, PROD_DROP)\n    prod_feat = prod.set_index(\"c\")[pf].add_prefix(\"prod_\")\n    prod_dg = prod.set_index(\"c\")[base_col].rename(\"dG_xtb_kcal\")  # baseline as XTB_DG\n    print(f\"product QM features: {len(pf)}\")\n\n    df = lab[[\"c\", \"SMILES\", \"dG_orca_kcal\"]].set_index(\"c\")\n    df = df.join(prod_dg, how=\"inner\").join(ald_feat, how=\"inner\").join(prod_feat, how=\"inner\")\n    df = df.dropna(subset=[\"dG_xtb_kcal\", \"dG_orca_kcal\"]).reset_index(drop=True)\n    print(f\"joined training rows: {len(df)}  total cols: {df.shape[1]}  \"\n          f\"(features={df.shape[1]-3})\")\n\n    DEST.parent.mkdir(parents=True, exist_ok=True)\n    df.to_parquet(DEST, index=False)\n    print(f\"wrote {DEST}\")\n    # quick target sanity\n    corr = df[\"dG_orca_kcal\"] - df[\"dG_xtb_kcal\"]\n    print(f\"Δ-target (dG_orca - dG_xtb): mean={corr.mean():.2f} std={corr.std():.2f} \"\n          f\"min={corr.min():.1f} max={corr.max():.1f}\")\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n",
>   "encoding": "utf-8",
>   "sha": "d856047d48554dc92aa369b21b0f839e2ba1d90a",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/pipeline/build_cb_training_table.py",
>   "display_title": "build_cb_training_table.py"
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch_file
>
> ```json
> {
>   "repository_full_name": "ElioChen/dG-prediction-of-aldehyde-to-benzoin-products",
>   "path": "data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ````json
> {
>   "content": "> **更正(同日晚些时候补充):** 下文第 6/7 节引用了项目此前的说法——多构象玻尔兹曼重标注能把误差下限从\"约\n> 1.57 降到约 1.38 kcal/mol\"。经核实,仓库中**没有任何数据支撑这个数字**。唯一真正跑完的试点实验\n> (`boltz_relabel_summary_20260626_1559.md`,n=95,每分子 5 个构象)得到的其实是**相反**的结果——换用重标注后的标签重新打分同一个冻结模型,MAE\n> 反而变差了(0.970→1.284),也就是说没有证据表明构象噪声在拉高误差下限。第 7\n> 节中原定规模的\"第三梯队\"**不应该**按这个前提启动;完整过程见 `descriptor-search-exhausted`\n> 记忆 2026-07-10 的更正记录。第一、二梯队的结论不受此更正影响。\n\n# 外部诊断文档 `benzoin_dG_project_context.md` 复核报告(2026-07-10)\n\n复核方式:本次会话直接核对仓库中的代码、数据文件与作业日志(而非直接采信文档自身的表述)。以下每一条结论都能对应到具体文件路径、报告或作业日志时间戳,均位于\n`/scratch-shared/schen3/benzoin-dg`。全文引用的冠军模型为\n`gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`——**测试集 MAE 1.503,RMSE\n2.257,R² 0.875**,275 个特征,不确定度分流 85%/15%(置信部分 MAE 1.252,转 DFT 部分 MAE\n2.923,ROC AUC 0.796)。\n\n## 0. 总体结论\n\n文档对\"模型在哪里出错\"(砜/磷子集)的判断是正确的,且已被独立验证。但其对\"为什么出错\"的机制解释(D4\n色散极化率外推,污染了整分子级 `P_int`)只是一个合理假设——而**文档列为最高优先级的具体修复方案(方案\nA,羰基反应位点的原子局部 `P_int`)其实已经在 2026-07-01 试过了,结果是零效应**。这不是一个应该优先尝试的新想法,而是文档作者不知情的一个已完成的阴性结果。文档中另外几项\"行动项\"同样,要么已经上线(不确定度路由,方案\nG),要么已经测试为零效应(几何假说、色散一致性)。真正尚未解决的问题比文档暗示的要窄得多,而针对\n~1.5 kcal/mol 误差下限**目前已知量化程度最高的最大杠杆,完全在文档讨论范围之外**(困难尾部分子的多构象玻尔兹曼重标注)。\n\n## 1. 文档中判断正确的部分\n\n| 主张 | 核实结果 |\n|---|---|\n| 砜/磷子集误差最大 | 已确认并量化:在误差最差的 15%(转 DFT)集合中,`ald_sulfonyl`/`prod_sulfonyl` 富集 **11.24×**,`has_P` 富集 **9.48×**(见 `REPORT_deep_error_analysis_champion275_20260707.md`) |\n| `P_int` 是 SHAP 最重要特征 | 已确认:275 个特征中排名第 1,mean\\|SHAP\\|=1.849(见 `REPORT_shap_mordredslim271_bdegxtb_20260707.md`) |\n| 标签(r²SCAN-3c)与 g-xTB 基线均使用 D4 色散 → 基线/标签间无色散不一致 | 判断正确,而且实际情况**比文档所说的更稳固**:几何优化阶段使用的 GFN2-xTB 本身也是 Grimme 课题组方法,其哈密顿量**原生内置 D4 色散**,g-xTB 同属该谱系。也就是说流程的三个阶段——几何优化、xTB 基线单点、DFT 标签——都共用 D4,而不仅仅是文档核对过的那两个阶段。 |\n\n## 2. 已被既有实验证伪的主张/建议\n\n### 方案 A——羰基反应位点的原子局部 `P_int`(\"最高价值,优先尝试\")\n\n**已于 2026-07-01 实现并测试过。** `cross_benzoin/analysis/add_morfeus_descriptors.py`\n正是文档所提议的做法——将 `morfeus.Dispersion.atom_p_int` 限制在反应位点(ketC/carbC/CHO_C),并附加逐原子的锥体化(pyramidalization)——`pipeline/analysis/finalize_correction_morfeus.py`\n在冠军 72 特征基线之上加入了这 9 个特征,使用相同的 70:20:10 划分与相同的集成模型:\n\n```\nbaseline_72(基线72特征):        MAE 1.574\n72_plus_morfeus9(+反应位点局部特征): MAE 1.576   (ketC/carbC/CHO_C 处的原子局部 disp_p_int + 锥体化)\n```\n\n差值(+0.002)落在实测噪声带以内(5 次重新划分种子的噪声带为 1.571 ± 0.013,见\n`REPORT_robustness_baseline72_20260702.md`)。**无论正向还是负向,均无可探测的效应。**\n(见 `REPORT_morfeus_augment_20260701.md`,作业号 24356687/24357492。)\n\n这并不能推翻背后的机制故事,但意味着文档给出的\"第一步该做什么\"其实并不是下一步——而是一个文档作者本应查到却没查到的阴性结果。若按文档原本的方案重跑一遍,只会浪费一轮计算。\n\n### \"次要问题:几何\"(GFN2 对 S=O/P 几何描述不佳 → 误差以几何形式伪装出现)\n\n现有的两条独立证据都指向另一个方向——即**电子结构描述**问题,而非几何问题:\n\n1. **`hbond-not-product-error-driver` 记忆(2026-06-19)的结论:** 在*固定*\n   xTB 几何下,富电子取代基(EWG)/含高价硫的苄偶姻产物侧大误差**并不**随氢键几何变化(相关系数\n   r = 0.06–0.15,R² < 0.025)。大误差分子恰恰是未形成分子内氢键的那些,原文明确得出\"这是\n   xTB 对含高价硫苄偶姻的电子结构描述问题,与氢键无关\"——也就是说,即使几何保持不变,误差依然存在。\n2. **几何质检数组中唯一跑完的一行**(`dftopt_36hard_v2/row_001.csv`,一个磺酰氟案例\n   `S(=O)(=O)F`):把几何从 GFN2 换成完整 r²SCAN-3c 重新优化后,DFT 单点能量仅变化\n   **−0.061 kcal/mol**(重原子位移 RMSD 0.335 Å)。这属于噪声量级,远非数 kcal 级别的修正。样本量\n   n=1,单独看不足以下定论——具体原因见第 3 节。\n\n综合来看:现有(尚不完整)的证据更支持\"xTB/g-xTB\n对高价硫/磷的*电子结构*描述有问题,几何只是次要贡献者\",这是一个**比文档\"色散+几何\"框架更窄、也更有证据支撑**的诊断,但并不完全等同于文档具体提出的\"P_int/D4\"故事——详见第 4 节的综合分析。\n\n### 方案 G——不确定度/保形预测/转 DFT 路由\n\n已经上线,不是待办事项。冠军模型包内置了分位数 XGB(5%/95%)预测区间、按区间宽度取最差 15%\n转 DFT 的路由标记,以及该不确定度是否真的追踪真实误差的实测 ROC AUC(0.796,置信 85% 部分\nMAE 1.252,对比转 DFT 15% 部分 MAE 2.923)。文档将其列为未来工作,应从清单中删除。\n\n## 3. 真正尚未解决的问题(真实存在的缺口)\n\n### 方案 C——S/P 子集的几何质检:基础设施已具备,但**卡在 4/36**\n\n`pipeline/compute/dft_opt_bench.py` 配合 `dftopt_36hard_v2` SLURM 数组作业(24300065 号,修复此前\nMPI 启动伪影问题后改为串行 nprocs=1,见 `dftopt-36hard-fair-rerun`\n记忆)是正确的工具,且已经搭建完成。但提交的 36 个高价 S/P/B/Si/硝基困难分子中,**只有 4\n个(`row_001/013/015/021.csv`)真正跑完**;其余 32 个日志文件为空(从未产出结果——很可能是超时或修复后从未重新提交)。汇总报告也从未生成过(`summarize_dftopt_36hard.py`\n脚本存在,但从未在 v2 数据上跑完过)。这是文档行动计划中**唯一**真正尚未完成的基础设施性工作,而不是一个已有定论的问题——在对几何问题下结论之前值得先补跑完。\n\n### 方案 B——将逐原子 D4 原始参数(C6、静态极化率、配位数 CN、EEQ 电荷)作为显式特征\n\n从未真正尝试过。有两个*相邻*的代理指标已测试为零效应——原子局部 `P_int`(上文,基于 SASA\n积分的色散势)以及 Multiwfn ADCH/QTAIM 逐原子电荷(在 2.5k 子集上 MAE 2.44→2.45,见\n`descriptor-search-exhausted` 记忆)——这降低了该方向成功的先验概率,但二者与 dftd4\n原始输出的 C6/极化率/CN/EEQ 并非同一物理量。另外需注意,ADCH/QTAIM\n的零效应结果只在 2.5k 分子的子集上测过,并未在完整 22 万分子库上验证(见\n`multiwfn-env-and-screen-gap` 记忆:全库筛选作业实际上从未加 `--multiwfn`\n参数运行,该数据在全库规模上根本不存在)。预期价值中低;成本不高,值得针对 S/P\n原子做一次窄范围定向测试(而非全量重新导出),但鉴于已有两个相邻零效应结果,不应作为优先项。\n\n### 方案 D——将高价态类别标志作为*模型*特征\n\n文档想要的基于 SMARTS 的标签(`ald_sulfonyl`、`ald_has_P`、`ald_imine`、`ald_amide`\n等)**已经存在**——它们在 `deep_error_analysis_champion275.py`\n中被计算出来用于富集诊断表格——但从未被加入 `finalize_correction_mordredslim271_bdegxtb.py`\n冠军模型的 `FEATS` 特征列表(该列表只有粗粒度的整分子布尔量\n`g_has_S`/`g_has_P`,没有 S=O/P=O 计数或氧化态代理量)。这是一个真实存在、尚未测试、且成本很低的缺口:打标签的代码和训练框架都已就绪,这只是合并特征列表并重新训练,不需要新的工程开发。\n\n### 方案 E——针对 S/P 子集单独计算 SHAP 及交互值\n\n尚未完成。目前已有的是全局 SHAP(基于全部 22\n万分子的前 25 个特征)和一个独立的标签富集分析(哪些*分子*属于困难样本)——但还没有人针对砜/磷/亚胺切片单独重新计算过\nSHAP,或提取 SHAP 交互值来看模型是否在尝试(且未能成功)把高价态子类单独区分出来,正如文档所提议的那样。成本很低——纯粹是对现有模型和现有测试集预测结果的再分析,不需要重新训练。\n\n### 方案 F——ΔE(电子)/ΔG(热力学)分解\n\n尚未完成。存在间接支持证据:**脂肪族 MAE 1.870 vs 芳香族 MAE\n1.327**(见 `REPORT_deep_error_analysis_champion275_20260707.md`)——可旋转键更多/柔性更强的分子误差更大,这与\nRRHO/低频振动模式熵误差是真实贡献因素的假设是一致的。但这在机理上与项目自身已经规划(但尚未启动)的困难尾部多构象玻尔兹曼重标注针对的是**同一个现象**——见第\n5 节。不应把方案 F\n当作一个独立的工作线;它是重标注工作的诊断依据,而不是替代方案。\n\n## 4. 综合分析:一个比文档更站得住脚的机理图景\n\n文档的因果链是:*D4 极化率参考流形不足 → `P_int`\n被扭曲 → S/P 上出现误差*。原子局部 `P_int`\n测试直接证伪了中间这一环——把 SASA 积分限制在反应位点并没有清理掉这个信号,所以\n`P_int` *特定的*色散机制并不明显是致命通路。结合氢键记忆中的独立发现(固定几何下的电子结构误差,而非\nNCI/色散效应)以及几何质检唯一一个数据点(几何弛豫对能量几乎没有影响),现有证据更站得住脚的解读是:\n\n> xTB/g-xTB 的**电子结构哈密顿量本身**(而不是特别针对其 D4\n> 色散项,也不主要是几何问题)是针对常规价态参考体系参数化的,因而普遍性地误描述高价硫(VI)/磷的成键情况——电荷分布、轨道能级、Wiberg\n> 键级(WBO)——而 `P_int` 只是*恰好相关*,因为砜基/含磷基团恰好也是体积大、极化率高、SASA\n> 占比大的取代基。专门去修复 D4/`P_int` 这条通路(方案 A、B)不太可能带来更多改善;这个误差更接近\"xTB\n> 系方法在高价态主族元素上的方法误差\",这更像是一个数据/标签问题(需要在该子类上做更多/更好的参考计算),而不是特征工程问题。\n\n这也与项目自身在 `descriptor-search-exhausted`\n中已有的结论一致(尝试了三个正交的描述符家族,误差下限均未移动):~1.5–1.57 kcal/mol\n的下限是**数据受限,而非特征受限**。\n\n## 5. 文档之外的额外发现:GNN+表格特征堆叠(stacking)悬而未决\n\n复核审阅的文档中未提及此项,是在核实\"不再深入尝试 GNN\"这一结论时发现的。2026-07-07 的\nGNN+表格特征堆叠测试(`gnn_dual_qm_champion275_ensemble.py`)在第三次尝试才成功跑完(前两次因\npickle/缓存相关的 bug 崩溃),产出了\n`ensemble_stack_champion275_20260707.csv`:在 w_gnn=0.35–0.4\n处的最佳混合给出 MAE 1.425,对比纯表格模型的 1.476(**−0.051**,表面上看是一个真实的提升)。**但是**,GNN\n测试集划分与表格模型测试集划分只在 21,911 个分子中重叠了 6,601 个(约\n30%)——这个对比是在一个部分的、不具代表性的子集上做的,而不是完整的留出测试集。正确的修复方案(`gnn_reeval_matched`,作业号\n24494009,按 ID\n匹配的完整重新评估)**在运行中途被 SIGTERM 取消,且从未重新提交**。这是一个真实存在、成本很低就能了结的悬而未决的线索:只需一次干净的\nGPU 作业即可确认或推翻这 −0.05 的堆叠增益是否真实。\n\n## 6. 是否真的需要新的模拟(新的 DFT/xTB 计算)?\n\n在本次复核涉及的全部内容中,只有以下两项真正需要新的物理模拟,而不是对已有数据的再分析:\n\n1. **补跑卡住的 `dftopt_36hard_v2` 数组**(36 个分子中有 32\n   个从未跑完)——成本低,基础设施已经搭好,能在 36 个高价 S/P/B/Si\n   困难样本上而不是目前 n=1 的孤例上,解决\"几何问题 vs\n   电子结构问题\"这一争议。这是文档方案 C 中真正合理的部分。\n2. **困难尾部分子的多构象玻尔兹曼重标注**——此前工作(见\n   `descriptor-search-exhausted`/`gxtb-dft-session-handoff`\n   记忆)已量化,该方法能将误差下限从约 1.57 降到约 1.38 kcal/mol(按平方和法估算),代价是**仅对尾部分子**做约\n   5 倍 DFT 计算量,而非对全部 22 万库都做。这是本项目迄今为止发现的单一最大杠杆,独立于且大于被复核文档中的任何方案,并且直接涵盖了文档方案\n   F(熵)的假设——目前尚未启动,按其自身的记忆记录,\"需要用户批准\"。\n\n其余所有可行动项(方案 D 高价态标志、方案 E 子集 SHAP、GNN\n堆叠的干净重跑、窄范围的方案 B 测试)都是对**已有数据**的特征工程/再分析——不需要新的\nDFT/xTB 计算。建议先把这些做完(成本低,几天而非几周),再决定是否投入约 5 倍\nDFT 成本的重标注计划,因为前者几乎零成本,而且可能会改变重标注计划本身应该瞄准的对象。\n\n## 7. 建议的执行顺序\n\n**第一梯队(无需新计算,数天内完成):**\n1. 对砜/磷/亚胺/酰胺切片单独计算 SHAP 及交互值(方案 E)。\n2. 把已有的 SMARTS 高价态标签作为真正的模型特征加入,重新训练,检查 MAE 是否在 S/P\n   子集上有可探测的移动(方案 D)。\n3. 对 GNN+表格堆叠做一次干净的、按 ID 匹配的重跑,了结 −0.05 这个疑问(见第 5 节)。\n\n**第二梯队(少量新计算,视第一梯队结果决定):**\n4. 补跑完 `dftopt_36hard_v2` 数组(剩余 32 个分子)——用真实样本量解决几何 vs 电子结构问题。\n5. 窄范围的方案 B 测试:仅针对 S/P 原子提取原始 D4 的 C6/极化率/CN/EEQ——前提是第一梯队的子集\n   SHAP 结果仍然指向该切片上与色散相关的特征。\n\n**第三梯队(真正的大杠杆,需要就成本明确批准):**\n6. 困难尾部分子的多构象玻尔兹曼重标注。\n\n## 8. 关于上传 GitHub\n\n已记录为后续任务,本次未启动。等第一梯队(以及可能的第二/三梯队)工作落地后,新仓库的自然内容应包括:`src/benzoin_dG/`、`pipeline/`(计算+分析+模型,剔除\n`__pycache__`/临时文件)、冠军模型的 `.joblib` 文件、关键的\n`REPORT_*.md`/`REPORT_*_zh.md` 报告对,以及 `FILE_MAP.md`\n索引。在真正推送之前,还需要你确认仓库名称、公开/私有,以及是否要包含完整的 22\n万行数据表(体积很可能超出普通 GitHub 仓库承受范围——建议用 Git LFS 或外部数据托管,仓库里只提交样例/schema\n数据)。\n",
>   "encoding": "utf-8",
>   "sha": "99e71a4cff228056f22f553c67cc0f23e13acf1a",
>   "display_url": "https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/data/cross_benzoin/homo_v6/viz_gxtb_20260625/REPORT_review_external_diagnosis_20260710_zh.md",
>   "display_title": "REPORT_review_external_diagnosis_20260710_zh.md"
> }
> ````
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `2024 2025 2026 bond dissociation energy prediction equivariant graph neural network benchmark ...`
> - Searched the web for `site:pubs.acs.org/doi "ANI-1xBB" bond dissociation energies ...`
>
> </details>
>
> </details>

</details>

可以把它发展成一个相当完整的子课题，但需要重新定义任务：不要把“给现有 ΔG 模型加四列 BDE/BDFE”继续放大，而应建立一个独立的、可与 ALFABET 公平比较的键解离数据集和模型体系，再研究这些预测是否能改善 homo/cross-benzoin 的 ΔG 与区域选择性。

我建议的题目方向是：

> 面向 homo-/cross-benzoin 化学空间的多保真、反应感知 BDE/BDFE 预测：从 2D 键中心图网络到 3D 等变反应差分模型

截至 2026-07-14，最合理的技术路线不是只选一个“最新网络”，而是同时建立三条相互补充的模型线：

1. ALFABET 类 2D 直接预测：最快、适合作为主基线；
2. 3D 等变反应差分模型：追求精度与构象/非共价作用；
3. MLIP 能量差模型：面向势能面、自由基和未来反应路径扩展。

## 一、先把子课题的科学问题限定清楚

建议设定三个相互关联但评价口径分开的任务。

### Task A：醛 formyl C–H BDE/BDFE

\[
\mathrm{RCHO}\rightarrow \mathrm{RC(O)^\bullet+H^\bullet}
\]

这是 aldehyde donor 活化能力和 acyl-radical 稳定性的近似表征。它适合：

- 与 ALFABET 的 C–H BDE 直接比较；
- 研究电子效应、芳香/脂肪差异；
- 作为 benzoin ΔG 模型的反应物侧描述符。

### Task B：benzoin 产物中心 C–C BDE/BDFE

\[
\mathrm{R^DCO-CH(OH)R^A}
\rightarrow
\mathrm{R^DCO^\bullet+\ ^\bullet CH(OH)R^A}
\]

这里 \(R^D\) 来自 donor 醛、最终位于 ketone 侧；\(R^A\) 来自 acceptor 醛、最终位于 carbinol 侧。仓库已有的 cross-benzoin 约定正是这个有序映射，见[cross-benzoin 架构说明](https://github.com/ElioChen/dG-prediction-of-aldehyde-to-benzoin-products/blob/main/cross_benzoin/ARCHITECTURE.md)。

这个任务比一般 C–C BDE 更有研究价值，因为中心键强度同时受到：

- 两个自由基片段的稳定化；
- donor/acceptor 角色不对称；
- 羰基与羟基的极化；
- 分子内 O–H···O=C 氢键；
- 两个取代基之间的空间、色散与位阻作用；
- 构象与立体化学；

影响。

### Task C：BDE/BDFE 对反应预测的增益

不要只问“BDE 预测得准不准”，还应问：

- 能否改善 homo-benzoin ΔG？
- 能否从 homo 迁移到 cross？
- 能否改善同一 donor 的 acceptor 排序？
- 能否区分 \(D+A\) 与 \(A+D\) 两个区域异构产物？
- 能否帮助不确定度路由和 OOD 检测？

Task C 是这个子课题区别于普通 ALFABET 复现的核心应用价值。

---

## 二、必须建立两套热力学标签，不能混为一谈

现有仓库的 g-xTB BDE/BDFE 与 ALFABET 标签不处于同一个参考体系。建议建立明确的双轨标签。

### 标签体系 1：ALFABET-compatible benchmark

用于公平比较：

- 气相；
- 298.15 K；
- 1 atm；
- homolytic dissociation；
- BDE：含 ZPE 的键解离焓；
- BDFE：RRHO Gibbs 自由能；
- 推荐尽量匹配 M06-2X/def2-TZVP。

这是扩展 ALFABET/BDE-db2 使用的定义：其数据包含 531,244 个 M06-2X/def2-TZVP BDE/BDFE，内部测试 MAE 约 0.61/0.60 kcal/mol。[BDE-db2 论文](https://pubs.rsc.org/en/content/articlepdf/2023/dd/d3dd00169e)

这套标签负责回答：

> 在相同 oracle 下，benzoin 专用模型是否优于通用 ALFABET？

### 标签体系 2：benzoin-project benchmark

用于项目本身：

- DMSO 溶液；
- 明确 1 M 标准态；
- 与反应 ΔG 的电子结构层级保持一致；
- BDE、BDFE 和电子能/热校正分开保存。

建议至少保存：

\[
\Delta E_\mathrm{diss},\quad
\Delta H_{298},\quad
\Delta G^\circ_{298,\mathrm{1M,DMSO}}
\]

不要再把“g-xTB 原始电子能”命名成标准 BDE 而不加限定。建议使用字段：

- `De_gxtb_dmso`
- `BDE298_dft_gas`
- `BDFE298_dft_gas`
- `BDFE298_dft_dmso_1M`

高质量标签可考虑：

- 几何和频率：r²SCAN-3c 或 ωB97X-D/def2-SVP；
- 单点：ωB97M-V/def2-TZVPPD、DLPNO-CCSD(T)/CBS 子集；
- 溶剂：SMD/CPCM(DMSO)；
- 低频：qRRHO；
- 自由基：波函数稳定性、\(\langle S^2\rangle\)、自旋污染和结构重排检查。

现有的约 22 万 g-xTB/GFN2 值非常适合作为低保真标签，但不应被当成最终高质量 oracle。

---

## 三、数据集设计：这是论文质量的决定因素

### 推荐的数据层次

| 层级 |                         规模建议 | 标签                   | 用途                  |
| ---- | -------------------------------: | ---------------------- | --------------------- |
| L0   | 现有约 22 万 homo + 可生成 cross | g-xTB/GFN2             | 预训练、低保真学习    |
| L1   |                    5,000–20,000 | 一致的中等 DFT         | 主监督训练            |
| L2   |                       500–2,000 | 高质量 DFT/复合方法    | Δ-learning、外部评估 |
| L3   |                          50–200 | DLPNO-CCSD(T)/实验可比 | 参考锚点              |

不建议一开始随机挑 10,000 个做 DFT。应采用“覆盖+困难样本”混合采样：

- scaffold/ECFP MaxMin；
- 芳香族与脂肪族分层；
- S、P、卤素、硝基、亚胺、酰胺、砜定向过采样；
- 分子量、柔性、环系分层；
- ALFABET 与 g-xTB 分歧最大的样本；
- 模型 ensemble 不确定度最大的样本；
- cross 中 donor/acceptor 电子性质极端不匹配的组合。

### cross-benzoin 不能随机按“产物行”划分

仓库把 cross pair 定义成有序的：

\[
\text{pair\_id}=\text{donor\_id}\_\_\text{acceptor\_id}
\]

因此必须建立二维冷启动测试：

| 测试集        | donor        | acceptor   | 研究问题           |
| ------------- | ------------ | ---------- | ------------------ |
| seen/seen     | 已见         | 已见       | 插值               |
| unseen/seen   | 未见         | 已见       | 新 donor           |
| seen/unseen   | 已见         | 未见       | 新 acceptor        |
| unseen/unseen | 未见         | 未见       | 真正组合外推       |
| homo→cross   | 仅 homo 训练 | cross 测试 | 是否学到可组合规律 |

划分流程应是：

1. 先按 aldehyde scaffold 将醛分组；
2. 再生成训练/验证/测试 pair；
3. 保证同一个醛的近同系物不会同时出现在训练和冷启动测试；
4. 将 \(D+A\) 和 \(A+D\) 放入同一大划分，防止区域异构体泄漏；
5. 额外设置官能团 OOD 和分子尺寸 OOD。

这比当前随机 70:20:10 更重要。

---

## 四、应该系统学习和比较的模型家族

## 1. ALFABET：必须保留的核心基线

扩展 ALFABET 是：

- 2D molecular graph；
- 六轮 message passing；
- bond embedding 最终直接回归 BDE/BDFE；
- 不需要构象或 QM 特征；
- 覆盖 C/H/N/O/S/F/Cl/P/Br/I；
- 内部测试 BDE/BDFE MAE 约 0.61/0.60 kcal/mol；
- C–C/C–H 分别约 0.77/0.74。

但真正 OOD 时误差可能升到 2–6 kcal/mol，再通过少量定向训练数据恢复。因此你的比较必须同时包含：

- 原始 pretrained ALFABET zero-shot；
- ALFABET 在 benzoin 数据上的 linear calibration；
- fine-tuned ALFABET；
- 从头训练的同架构模型。

否则无法判断差异来自架构还是训练域。

## 2. D-SPOC + LightGBM：实验相关基线

2024 D-SPOC 将断键前后结构和理化性质的变化显式编码，再用 LightGBM 预测，结合计算和 iBonD 实验数据，测试 MAE 约 1.03 kcal/mol。[D-SPOC 论文](https://onlinelibrary.wiley.com/doi/full/10.1002/cjoc.202400049)

它非常适合加入，因为：

- 与你现有的 XGB/Mordred 工作流接近；
- 可以判断深度学习的增益是否真的来自架构；
- differential descriptors 与“母体→两个自由基”的物理过程一致；
- 在中小规模高质量数据上往往比从头训练大 GNN 更稳。

建议构建 benzoin-specific D-SPOC：

\[
x_\mathrm{diff}
=x(A^\bullet)+x(B^\bullet)-x(AB)
\]

包含：

- Morgan/Mordred 差分；
- 元素、键型和局部环境变化；
- 形式电荷、自旋、多重度；
- 分子量、SASA、极化率；
- 片段自由基稳定性代理量。

## 3. BonDNet：反应差分基线

BonDNet 不只看母体，而是显式编码母体与解离产物之间的图差：

\[
h_\mathrm{rxn}
=h(A^\bullet)+h(B^\bullet)-h(AB)
\]

它可处理带电分子以及均裂/异裂，公开结果约 0.022 eV。[BonDNet 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8179073/)

对于 benzoin 中心 C–C BDE，它比 ALFABET 的归纳偏置更符合物理，因为两个自由基片段的稳定化正是决定 BDE 的核心。

## 4. 现代 2D 键中心模型

推荐构建一个实用的强 2D 主模型：

- directed message-passing；
- 明确标记待断键；
- bond-centered readout；
- 局部子图 + 全局分子 pooling；
- 多任务同时预测 BDE、BDFE、低保真值和高保真校正；
- ensemble + conformal UQ。

可使用的输入：

- 原子：元素、价态、杂化、芳香性、环、形式电荷、H 数；
- 键：键级、共轭、环、立体信息；
- 断键标记；
- 到断键的拓扑距离；
- donor-side/acceptor-side fragment 标记；
- homo/cross mask。

这是最公平的 ALFABET 后继模型，因为仍只需要 SMILES。

## 5. 3D 等变反应差分模型：建议作为论文主模型

推荐主架构可暂命名为 `BenzoinBondNet-3D`：

```text
母体/产物多构象 ── E(3) 等变编码器 ──┐
自由基片段 A• ─── E(3) 等变编码器 ───┼─ reaction-difference readout
自由基片段 B• ─── E(3) 等变编码器 ───┘
                           +
       新断键局部表示 / through-space 表示
                           ↓
             BDE、BDFE、Δ-correction、UQ
```

候选 backbone：

- PaiNN：较轻、适合作为首个 3D 基线；
- DimeNet++/GemNet：角度信息强；
- MACE：多体和势能表示较强；
- NequIP/Allegro：适合进一步发展成 MLIP；
- EquiformerV2 类模型：资源充足时再考虑。

反应表示建议为：

\[
h_\mathrm{rxn}
=h(A^\bullet)+h(B^\bullet)-h(AB)
\]

再与断键局部表示 \(h_\mathrm{bond}\) 和全局表示 \(h_\mathrm{global}\) 拼接。

### 为什么需要多构象

benzoin 中心 C–C BDE 可能明显依赖：

- O–H···O=C 氢键；
- 两个芳环/取代基的空间接近；
- 扭转角；
- 位阻释放；
- 自由基片段的构象重排。

建议每个物种保留 3–10 个低能构象，用以下任一方式聚合：

\[
G=-RT\ln\sum_i e^{-G_i/RT}
\]

或用 attention pooling 学习构象权重。

2026 年的 OLED 双注意力工作表明，将 through-bond 与 through-space 注意力和色散矩阵结合，可显著改善受长程分子内非共价作用控制的 BDE。[Chemistry of Materials 论文](https://pubs.acs.org/doi/10.1021/acs.chemmater.6c01241)

对于 benzoin，这比单纯堆积更多 Mordred 描述符更有潜力。

## 6. MACE/MLIP 能量差路线

MACE 的思路不是直接回归 BDE，而是分别预测母体和自由基的能量：

\[
D_e=E_\mathrm{MLIP}(A^\bullet)+E_\mathrm{MLIP}(B^\bullet)
-E_\mathrm{MLIP}(AB)
\]

在 CYP 脂肪族 C–H 测试中，MACE 得到 1.37 kcal/mol RMSE，并在 BDE 排序上优于 ALFABET、AM1 和 GFN2-xTB。[MACE-BDE 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10782450/)

它的优势是：

- 保持能量的可加性和广延性；
- 可做几何优化；
- 可获得平滑拉键曲线；
- 可进一步研究 H abstraction 或 C–C 断裂路径。

但它的风险也最大：

- 必须正确覆盖开壳层自由基；
- 近平衡训练数据不保证断键外推；
- BDFE 仍需频率/采样；
- C–C 断裂可能产生显著多参考特征。

所以建议它是第三条研究线，不是第一版模型。

2025 年的相关进展包括：

- ANI-1xBB：显式加入 off-equilibrium bond-breaking 构型，改善 BDE、势垒和反应路径。[ANI-1xBB](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c00347)
- Morse 势物理约束数据增强：改善解离曲线与 BDE。[PCDA](https://pubmed.ncbi.nlm.nih.gov/39874212/)
- atomization-energy constraint：通过约束孤立原子能量基准，提高 MLIP 的 BDE 与反应能表现。[HIP-NN-AE/ANI-AE](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00341)

这些说明，若发展 MLIP，必须显式处理能量零点、孤立原子/自由基和拉键构型，不能只拿通用 MACE 在现有几何上微调。

---

## 五、homo/cross 描述符该如何处理

这里需要区分两个问题。

### 预测“产物中心 C–C BDE”时

BDE 是产物及两个均裂片段的固有性质。理论上，模型只需要：

- 带标记中心 C–C 键的产物；
- 或产物 + 两个自由基片段。

不应强制依赖反应物的 donor/acceptor 描述符，否则同一产物由于生成路径命名不同可能得到不同 BDE。

推荐输入：

\[
x_\mathrm{BDE}=
[x_\mathrm{product},
x_\mathrm{new\ bond},
x_{\mathrm{frag1}},
x_{\mathrm{frag2}}]
\]

其中两个自由基片段的合并对交换应保持不变：

\[
h_\mathrm{frags}
=h_1+h_2,\quad |h_1-h_2|,\quad h_1\odot h_2
\]

这样满足均裂产物交换对称性。

### 将 BDE 用于 homo/cross ΔG 时

这里 donor/acceptor 角色是有序的，不能对称池化。应采用：

\[
z_\mathrm{pair}=[
z_D,\ z_A,\ z_D-z_A,\ z_D\odot z_A,
z_P,\ z_\mathrm{newCC},\ \mathrm{BDE}_{CC}
]
\]

必要特征至少分三组：

#### Donor 醛特征

- formyl C–H BDE/BDFE；
- carbonyl C/O 电荷、Fukui、WBO；
- HOMO、亲核性/电离能；
- acyl-radical 稳定化；
- 位阻与构象。

#### Acceptor 醛特征

- carbonyl C electrophilicity/Fukui+；
- LUMO；
- C=O WBO；
- 局部电荷；
- carbonyl 位阻。

#### 产物/相互作用特征

- 中心 C–C BDE/BDFE；
- 新 C–C WBO 和键长；
- ketone/carbinol 两侧局部电荷；
- O–H···O=C 距离、角度和能量代理；
- 关键二面角；
- through-space 接触；
- 色散/非共价相互作用；
- 构象能隙或构象熵。

仓库现有的 `_ketC/_ketO` 与 `_carbC/_hydO/_hydH` 命名非常适合保留；关键是不要把 donor 与 acceptor 特征合并成无序总和。

---

## 六、推荐的模型实验矩阵

| 编号 | 模型                    | 输入             | 主要目的       |
| ---- | ----------------------- | ---------------- | -------------- |
| B0   | ALFABET zero-shot       | SMILES + bond    | 通用基线       |
| B1   | ALFABET fine-tune       | SMILES + bond    | 域适应         |
| B2   | ECFP/Mordred + LightGBM | 2D 描述符        | 传统强基线     |
| B3   | D-SPOC-LightGBM         | 母体/片段差分    | 可解释反应基线 |
| B4   | bond-centered D-MPNN    | 2D marked graph  | 主要快速模型   |
| B5   | BonDNet-style           | 母体+自由基图差  | 反应差分基线   |
| B6   | PaiNN/MACE direct head  | 3D 多构象        | 3D 性质模型    |
| B7   | 3D reaction-difference  | 母体+两个自由基  | 建议主模型     |
| B8   | MLIP energy difference  | 能量/力/拉键数据 | 势能面模型     |
| B9   | 2D+3D+QM stacking       | 上述组合         | 最终性能上限   |

每个模型都应同时测试：

- 仅 homo；
- homo 训练→cross 测试；
- homo+cross 联合训练；
- 新 donor；
- 新 acceptor；
- 双新组分；
- 芳香/脂肪；
- S/P/卤素等 OOD。

---

## 七、训练目标建议采用多任务和多保真

推荐损失函数：

\[
\mathcal L =
w_1 L(D_e^\mathrm{low})
+w_2 L(\Delta D_e^\mathrm{high-low})
+w_3 L(BDE_{298})
+w_4 L(BDFE_{298})
+w_5 L(\Delta G_\mathrm{benzoin})
+w_6 L_\mathrm{rank}
\]

其中：

- 低保真头学习现有 g-xTB 数据；
- Δ-head 学习 DFT−g-xTB；
- BDE/BDFE 多任务共享结构表示；
- 下游头学习 benzoin ΔG；
- ranking loss 用于 donor/acceptor 排序和区域选择性。

这比单独训练一个 BDFE 回归器更有效，因为仓库已有证据表明 BDFE 与其他描述符高度冗余，而 BDE 携带更多独立信号。

还可以把 ALFABET 作为 baseline：

\[
\hat y =
y_\mathrm{ALFABET}
+\Delta_\theta(x)
\]

这样论文问题变成：

> benzoin-specific 3D、片段和溶剂信息能在通用 ALFABET 之上修正多少？

这是一个非常清晰的比较。

---

## 八、评价指标不能只报告随机测试 MAE

### 键能指标

- MAE、RMSE、median AE；
- 90/95/99 percentile error；
- \(R^2\)；
- calibration error；
- 官能团分层 MAE；
- 芳香/脂肪分层；
- 分子尺寸外推；
- 与实验或高层理论的误差。

### 排序指标

- Spearman \(\rho\)；
- Kendall \(\tau\)；
- 同一 donor 下 acceptor 排序；
- 同一 acceptor 下 donor 排序；
- \(D+A\) 与 \(A+D\) regioisomer 排序；
- top-k 候选召回率。

### 下游指标

分别比较 ΔG 模型：

1. 不含 BDE；
2. 真实低保真 BDE；
3. ALFABET BDE；
4. fine-tuned ALFABET；
5. 2D 专用模型；
6. 3D 专用模型；
7. DFT BDE；
8. BDE+BDFE。

需要配对 bootstrap 和多随机种子置信区间。现有 1.525→1.503 的改进只有 0.022 kcal/mol，必须证明其跨划分稳定性。

---

## 九、建议的实际实施顺序

### Phase 1：定义与基线，约 1–2 个月

- 整理 aldehyde C–H 和 product C–C 数据表；
- 明确气相/溶液标签字段；
- 检查自由基生成和原子映射；
- 在现有数据上运行 ALFABET；
- 训练 ECFP/Mordred、D-SPOC 和 bond-centered D-MPNN；
- 建立正确的 cross 冷启动划分。

这一阶段不需要大规模新 DFT，就能判断该子课题是否有明显信号。

### Phase 2：中等规模高质量标签，约 2–4 个月

- 选择 5k–20k 代表样本；
- 生成一致 DFT BDE/BDFE；
- 训练 multi-fidelity Δ-model；
- 完成 ALFABET zero-shot/fine-tune 公平比较；
- 分析 homo→cross 迁移。

### Phase 3：3D 主模型，约 3–6 个月

- 多构象 PaiNN/MACE；
- parent+fragment reaction-difference；
- through-space/氢键注意力；
- 2D/3D stacking；
- 构象消融。

### Phase 4：反应型 MLIP，可作为后续扩展

- 拉键构型；
- 原子化约束；
- Morse/PCDA；
- 开壳层训练；
- 中心 C–C 解离曲线；
- 与 E3D emergent bond representation 对照。[2026 E3D 论文](https://www.nature.com/articles/s41467-026-74919-8)

---

## 十、我最推荐的“最小可发表版本”

如果资源有限，不要一开始训练大型通用 MLIP。最小但具有论文价值的版本是：

1. 建立 aldehyde C–H 与 benzoin C–C 的 homo/cross 专用数据集；
2. 建立严格的 donor/acceptor 双冷启动划分；
3. 比较 ALFABET、D-SPOC、bond-centered D-MPNN、BonDNet-style；
4. 构建多构象 3D 等变 reaction-difference 模型；
5. 使用 g-xTB→DFT Δ-learning；
6. 证明模型能否改善 cross-benzoin ΔG 或区域选择性；
7. 对 S/P、芳香/脂肪和氢键子集给出机制解释。

最有希望的主模型不是“更多 Mordred + XGB”，而是：

> ALFABET/2D 表示作为快速先验 + 多构象 E(3) 等变 parent–fragment reaction difference + g-xTB→DFT 多保真校正 + role-aware homo/cross 下游编码。

这既继承现有仓库的大规模低保真数据优势，也真正吸收了目前 BDE 研究中最重要的进展：反应差分、3D 等变性、多构象、长程非共价作用、物理能量约束和 OOD 域适应。
