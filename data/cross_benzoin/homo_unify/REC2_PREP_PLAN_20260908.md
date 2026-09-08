# Rec-2 (homo+cross 统一模型) — 准备阶段计划

**日期**: 2026-09-08 · **决策**: 用户同意「Tier B 并行准备统一表，暂不重训」。
**关联**: `data/analysis/homo_cross_gap/FINDING.md` §Task C（verdict AMBER-GREEN，naive_merge +0.11 kcal）·
`HANDOFF_20260908.md` §1.5 · memory `next-work-three-recommendations`（建议 2）。

## 现状盘点（09-08 核对）

| 组件 | 状态 |
|---|---|
| homo 可用标签 | `homo_unify_v1` = **30,000 对**（`homo_unify_v1_dft.csv`：id, dG_orca_kcal）。**不是 220k**——其余在 2026-07 purge 中丢标签 |
| homo 产物 QM 块（mulliken/wbo/fukui/qtaim/adch/sterimol/vbur…） | ✅ `homo_unify_v1_products.csv` 72 列已含 |
| homo 产物 g-xTB BDE (`bde_gxtb_kcal`) | ✅ `homo_unify_v1_bde.csv` 30,000 行已存 |
| homo donor/acceptor 醛描述符（醛 QM + slim102 mordred + rdkit，260 里 153 列） | ✅ 从共享 220k 库 join（`homo_v6/aldehydes_all.csv` + `aldehydes_mordred_slim102.csv`）；homo 中 donor==acceptor |
| **homo 产物 2D/3D mordred**（260 里约 120 列，`product_mordred_*`） | ❌ **缺** `homo_v6/products_mordred_descriptors.csv` |
| homo 产物几何 | 归档在 `data/cross_benzoin/bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst`（2200 chunk，每 ~199 个 `xyz/pNNNNNN.xyz`，GFN2-opt，09-02 BDE featurize 产）。**30k 中 24,569 有归档路径（82%）**，5,431 缺（errored / 不在 clean products_all）|

## 准备步骤（label 无关，可与 Tier B 并行；约半天，纯 CPU）

1. **id→归档 xyz 映射**：已生成 `scratchpad/homo_unify_xyz_map.csv`（来自 `homo_v6/products_all.csv` 的 `id,xyz_file`，24,569/30,000 命中）。
2. **解档产物几何**：按 chunk 从 `geom.tar.zst` 抽出 homo_unify 那 24.6k 个 `pNNNNNN.xyz` 到 scratch。
3. **产物 mordred**：`cross_benzoin/add_mordred_cross_products.py`（TARGET_MODULES = MoRSE/CPSA/Polarizability/GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA，`ignore_3D=False`，~1s/分子）→ 分片阵列 → concat → `homo_v6/products_mordred_descriptors.csv`。
   - 缺几何的 5,431 行：先 median-impute（predict_dg 对新对同样处理），或后续补一个小 geom-regen 阵列。
4. **组装统一表**（改 `assemble_cross_training_table_unified_v2.py` 或新写 v3）：`--rounds 1..10` + slim260 + append `homo_unify_v1` + 加 `is_homo` 标志 + `sample_weight` 列（homo 降权到 homo:cross ≈ 1:1，现成比例 30k:23k 已接近，权重 ~1.0）。骨架 split 用 `homo_unify_v1_scaffold_split_lookup.csv` + cross 的 `new_scaffold_split`。
   - 输出 `data/cross_benzoin/homo_unify/cross_train_table_r1_10_homo_unified_slim260.parquet`。**到此为止，不重训。**

## Tier B 落地后（post-merge）

5. 用**新自洽 B97-3c 标签 + `BASELINE_COL=dG_b973c_kcal` + schema v2 (257)** 重跑步骤 4 的组装（换 label/baseline 列即可，特征侧复用）。
6. `train_scaffold_disjoint.py` 重训 champion ensemble + GNN，带 `is_homo`/`sample_weight`，对比 `cross_only` vs `naive_merge` 在 cross 骨架不相交 holdout (n=448)。
7. 判据：−0.11 kcal 方向在 260-feat + 新基线下是否存活。存活 → 再投入 GNN homo-pretrain→finetune 路径。不存活 → 记录，关闭 Rec-2。

## 已知 caveat

- 归档产物几何是 09-02 BDE featurize 的 GFN2-opt，未必与 cross rounds 的 funnel_v3+ohess 产物几何逐一同级；mordred 3D 形状描述符对 opt 级别较鲁棒，但作为 caveat 记录。
- 18% 产物无归档几何 → 那部分 `product_mordred_*` 走 impute，homo 贡献略打折。
- 收益软：< 1 bootstrap SE @ n=448；72-feat 代理比冠军弱 0.5 kcal；Tier B 若把冠军推到 ~1.0-1.5，+0.11 可能被噪声淹没。

---

## 附：全 219k homo label 恢复 —— 议程项（未排期）

purge 丢失的 homo `dG_orca_kcal` 标签（219k 中仅 30k 幸存于 `homo_unify_v1`）。恢复向量待查：
- 部分 DFT-SP 归档 / round tarball（参照 cross 侧 r8/9 label 恢复经验，memory `benzoin-dg-repo-moved-and-data-loss` 的 `cv_predictions.csv` 恢复技巧）
- `/gpfs/home4/schen3/benzoin_backups/` home 归档（memory `home-backup-archives-recovery`：曾 bit-exact 恢复醛 mordred + BDE）
- `data/raw/dft_sp_funnelv3/dft_labels_all.parquet`（`build_homo_for_unification.py` 引用了「219,364 valid r2SCAN-3c labels」——该文件当前**不在** restored repo，需找归档）
- 若标签物理丢失且无归档 → 评估重算成本（219k × r2SCAN-3c SP，复用归档几何则 SP-only）

优先级：低于 Tier B 主线；可在 Tier B / Rec-2 prep 之后作为下一条长跑候选。

---

## 09-08 更新：用户要求补**全 220k**（非 30k），CSV 完整（label + 描述符 + mordred），xyz 保持压缩

### fat 节点成本
`TRESBillingWeights`：fat_rome/fat_genoa `cpu=1.5`，rome/genoa `cpu=1.0` → **fat 贵 50%/CPU-h**（换 4.3× 内存）。

### 全库补全 —— 盘点结果（09-08）

| 资产 | 全库状态 | 动作 |
|---|---|---|
| 醛 QM 描述符 `aldehydes_all.csv` | ✅ 209,527 行 | 有 |
| 醛 mordred `aldehydes_mordred_slim102.csv` | ✅ 220,524 行 | 有（另 home 备份 `mordred_aldehydes.tar.gz` 2.3G 全量 1826 列） |
| 醛 BDE (alfabet + bdfe_gxtb) | ✅ 220,522 行 | 有 |
| 产物 QM 描述符 `products_all.csv` | ✅ 184,199 行 | 有 |
| 产物 BDE (alfabet + bdfe_gxtb) | ✅ ~219k 行 | 有 |
| **产物 mordred** | ✅ **在 home 备份** `homo_v6_scratch_archive/mordred_products.tar.gz`（2.4G，2196 chunk，1826 列，覆盖全部 53 个 champion `product_mordred_*`，0 缺）| **恢复中**：job 26476123（genoa，<1h）→ `products_mordred_full.parquet` + `products_mordred_descriptors.csv` |
| 产物几何 xyz | ✅ 归档 `bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst` + home `homo_product_chunk_geoms_20260902.tar` 568M | 保持压缩（用户：xyz 可压缩） |
| **DFT 标签 `dG_orca_kcal`（全 219k）** | ❌ **未找到**：`dft_sp_funnelv3/dft_labels_all.parquet`（build_homo_for_unification.py 引用的 219,364 标签）不在 restored repo，**不在任何 home 备份 tar**（全部 home 备份只含 cross 的 `dft_sp_cross/`）。仅 30k 幸存于 `homo_unify_v1_dft.csv` | **需重算** —— 见下 |

### DFT 标签全库恢复方案（待用户确认，大工程）

标签物理丢失。可行路径 = 重算，与 cross Tier B 同机制（`rec1_b973c_tierB_worker.py` 可指向 homo 对）：
- homo dG = G(product) − 2·G(aldehyde)；需 r2SCAN-3c/CPCM(DMSO) SP：~184k 产物 + ~209k 醛 ≈ **~400k SP**
- 若做**自洽 B97-3c**（与 post-Tier-B cross champion 可池化，推荐）：每物种 funnel_v3+ohess 几何 + r2SCAN + B97-3c + gxtb SP ≈ 9 CPU-h → **~2M+ CPU-h，~3-4 周**战役（量级≈ cross Tier B）
- 若只 r2SCAN-3c SP（复用归档几何，非自洽）：~5 CPU-h/SP → ~2M CPU-h，仍是周级
- **排序建议**：cross Tier B 落地并验证 → 再起「homo 自洽 B97-3c relabel」作为下一个大战役（同 worker、同标签方案）。现在先把便宜的描述符/mordred 补齐，标签留 30k。

### 剩余低成本补全步骤（现在做）
1. 恢复产物 mordred（job 26476123）✅ 进行中
2. 恢复醛全量 mordred（`mordred_aldehydes.tar.gz`）—— 若 `aldehydes_mordred_slim102.csv` 不够用
3. 校验四件套 id 对齐（醛/产物 × QM/mordred/BDE），产出一张 `homo_v6/LIBRARY_MANIFEST.md` 记录每个文件行数/覆盖率/id 规范
4. 待 26476123 完成 → 更新 assembler 走全 220k（有标签的行才进训练，其余作候选/推理库）
