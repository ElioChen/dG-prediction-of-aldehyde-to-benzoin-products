# BDE 阵列完成后的模型 sweep 计划 / Post-array model sweep

> 触发条件:featurize array **`26316404`** (`bde_homoprod`, genoa, 2209 tasks) 全部完成。
> 本文件 = HANDOFF_20260902 §2.2 的展开版,把"重训 B6"扩成一个更宽的 model sweep
> (用户 2026-09-02 指示:继续推进、提交更多任务、尝试更多模型)。
> Monitor task `bm8632d2g` 会在阵列出队时发通知。

---

## 0. 前置:装配描述符库 + 校验几何归档 / assemble + verify

```bash
cd /gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python

# 0a. 装配两个库(会覆盖 partial 42k aldehydes_all.csv)
$PY pipeline/bde/assemble_homo_descriptor_libs.py \
    --chunk-dir data/cross_benzoin/bde_homo_product_featurize_20260902
#  -> data/cross_benzoin/homo_v6/{products_all.csv, aldehydes_all.csv}
#  期望:aldehydes_all 从 42,335 行 -> ~220k;products_all ~新建 ~205k

# 0b. 几何归档收尾 —— 由 staging job 26324608 (archive_completed_chunk_geoms.sh) 自动做:
#     每个完成的 chunk 的 xyz/ + ald_xyz/ 已打成 chunk_XXXX/geom.tar.zst 并删散文件。
#     阵列出队后它做最后一遍 sweep,再把所有 geom.tar.zst 合并成
#     ~/benzoin_backups/bde_homo_rebuild_20260902/homo_product_chunk_geoms_20260902.tar
#     然后自行退出。**确认它退出且 .geom_archived 数 == chunk 数** 再继续。
#     若它没跑(比如 staging 排不上),手动:
#       bash pipeline/bde/archive_completed_chunk_geoms.sh   # 会一次性补齐并退出
#     这一步取代了 HANDOFF §2.2.2 的 `rm -rf .../xyz`(用户要求压缩保留几何,不是删)。

# 0c. 备份新装配的库
BK=~/benzoin_backups/bde_homo_rebuild_20260902
tar -C data/cross_benzoin/homo_v6 -czf $BK/products_all_20260902.csv.tar.gz products_all.csv
tar -C data/cross_benzoin/homo_v6 -czf $BK/aldehydes_all_full_20260902.csv.tar.gz aldehydes_all.csv
sha256sum data/cross_benzoin/homo_v6/{products_all,aldehydes_all}.csv >> $BK/SHA256SUMS.txt
```

---

## 1. 模型 sweep(提交顺序,全部 scaffold-disjoint 划分)

| # | job / 脚本 | 分区 | 内容 | 依赖 | 期望 |
|---|---|---|---|---|---|
| 1 | `sbatch pipeline/slurm/submit_b6_scaffold_disjoint_ckpt.sh` | gpu_h100 ×2 | B6 冠军带 checkpoint 重训(醛+产物) | 0a | 醛 MAE ~1.58 / 产物 ~3.06;产出可部署 `.pt` |
| 2 | `sbatch pipeline/slurm/submit_b6_deep_ensemble.sh` | gpu_h100 ×10 | **新** B6 5-seed deep ensemble(醛+产物),给 route-to-DFT 的 epistemic σ | 0a | ensemble MAE ≤ 单 seed;σ~\|误差\| spearman > 0 |
| 3 | `sbatch pipeline/slurm/submit_b4_b5_scaffold_disjoint.sh` | gpu_h100 ×4 | B4(纯 2D D-MPNN)+ B5(BonDNet 式)scaffold-disjoint 重训,**补齐诚实数字**(现表里 B4 产物侧仍是 naive-split;B4/B5 从没存过 checkpoint) | 0a | B4 醛 ≳1.6 / 产物 ≳3.6;确认 B6 的 `x_d` 融合优势在诚实划分下仍在 |
| 4 | `$PY pipeline/bde/gbm_bakeoff_hspoc.py --which {aldehydes,products} --split-file … --out …` | 本地/staging CPU | XGB vs HistGBM vs RF vs ExtraTrees,H-SPOC 特征全量重跑(pre-array 已在 partial 42k 上跑过一次做筛选) | 0a | 决定全量下最好的 cheap-CPU head |
| 5 | `sbatch pipeline/slurm/submit_gnn3d.sh` (+ `submit_gnn3d_nequip.sh`) | gpu_h100 | Phase-3 3D 反应差分模型(SchNet/DimeNet,`pipeline/analysis/gnn3d_schnet_dimenet.py`)。**几何现在保留在 chunk_XXXX/geom.tar.zst,先解包**到一个平铺目录 | 0a + 0b + 解包几何 | 看 3D 几何模型能否超过 B6 的 `x_d` 融合 |

### sweep 后

```bash
# 6. 聚合 deep ensemble
$PY pipeline/bde/aggregate_b6_ensemble.py \
    --ens-dir runs/logs/scaffold_disjoint_bde/ensemble
#  -> ensemble/{aldehydes,products}_ensemble_pred.csv + ensemble_summary.json
#     (含 route-to-DFT 覆盖率-vs-kept_MAE 曲线)

# 7. 所有结果入库(gitignore 排除 *.pt / runs/,必须 -f)
git add -f runs/logs/scaffold_disjoint_bde/**/*.json \
           runs/logs/scaffold_disjoint_bde/**/*_pred.csv \
           runs/logs/scaffold_disjoint_bde/**/*.pt
# 并把这些路径加进 cross_benzoin/slurm/submit_backup_recovery_artifacts.sh 的归档清单
#   (purge 教训:gitignore + 无 home 备份 = 一次策略变更就没)

# 8. 更新 pipeline/bde/STATUS.md 第二节排名表 + 第六节状态;跑 build_hard_set.py 补 BDE source
$PY pipeline/analysis/build_hard_set.py
```

---

## 2. 更长线的候选(sweep 有结论后再定,勿盲目提交)

- **B6 + best-GBM 堆叠**:B6 图嵌入 + H-SPOC GBM 残差,promote_gnn_stacking 那套思路搬到 BDE。
- **Δ-learning 头**:预测 (DFT − g-xTB) 的 BDE 修正 —— 但目前没有规模化的 DFT BDE 标签
  (只有 dft_bde_pilot n=100 + arbitration round2 n=128),先等标签。
- **substructure → route_to_dft veto 规则**(零训练):P=O / 鏻盐 / 多磺酰基 / 多酰胺 /
  nitro 且 bde_gxtb>100 → 推理时强制 DFT。见 `notes/gxtb_baseline_failure_hardtail_20260902.md`。
  可以现在就写进 `predict_bde_champion.py`,和 deep-ensemble σ 双保险。
- **homo→cross 迁移学习曲线补点**:等窗口 B 的 round8/9/10 cross BDE 数据齐了再做(§STATUS 三.2)。

---

## 3. 资源纪律

- 窗口 A 的 GPU sweep 用 **gpu_h100**;不要在 `26316404` 还在跑时提大 genoa 阵列。
- geom 归档用 **staging**(job 26324608),纯 I/O,不占算力配额。
- scratch inode:阵列会加 ~45 万 inode,但增量归档同时在回收 —— sweep 前确认
  `find …_20260902 -maxdepth 2 -name .geom_archived | wc -l` 接近 chunk 数。
- push:等窗口 A + B 的东西都收齐,由收尾的窗口统一 push `agent/recovery-20260902`。
