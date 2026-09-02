# benzoin-dg 项目状态与开放问题(2026-09-02)

_一次性快照,回答 5 个具体问题:待补算任务、磁盘/inode 安全、round11 必要性、
各任务的最优模型、需要长期记录的事项。配套:根 `STATUS.md`(生产路径,较旧)、
`pipeline/bde/STATUS.md`(BDE 子课题)、`cross_benzoin/docs/STATUS_{EN,ZH}.md`(dG 主线,
较旧但有 2026-07-17 勘误)、`RECOVERY_REPORT_20260902.md`(purge 恢复)。_

> **项目框架(用户 2026-09-02 明确)**:主线是 benzoin 反应 ΔG 的预测 / 主动学习,
> **homo 和 cross 都是同一个项目的一部分**;BDE 预测是给 ΔG 模型供描述符的子线。

---

## 1. 还需要补算 / 重算的任务与分子

### 1.1 正在跑(本会话发起)
| 任务 | job | 规模 | 状态 / ETA |
|---|---|---|---|
| **BDE homo 描述符库重算**(`products_all.csv` + `aldehydes_all.csv` 全量) | `26316404` genoa `bde_homoprod` | 220,859 homo pair,`featurize_product.py --multiwfn --emit-aldehydes`,2209×CHUNK100 | 🟢 ~37/2209 完成(17:15),单任务中位 45min、最长 90min;genoa 94% 占满,实际有效并发 <96 → **ETA ~1.5–2 天(2026-09-03 晚 ~ 09-04)** |
| ↳ 之后 `assemble_homo_descriptor_libs.py` → B6 scaffold-disjoint 重训(醛+产物,带 checkpoint) | — | 2×H100 | array 完成后 ~15min + ~3h + 排队 |

### 1.2 dG 主线 round10(并行会话正在处理,本会话不插手)
- round10 fat20 stage1 特征化 **已完成**(16,000 product + 醛描述符)。
- 组装进行中:`cross_round10_fat20_stage1_features_..._products_merged.parquet`(75MB,
  16:24 生成)、`mordred_products/`、`bde_gxtb/`(163 chunk)——**有并行 agent 在做
  round10 的 product mordred + product bde_gxtb + 合表**(`bde_cross`/`mordred_cross`
  job 最近完成)。醛侧的 bde_gxtb / mordred slim102 已由 purge 恢复的全库文件覆盖,
  不用重算。**本会话不推进 round10 组装线**,避免和并行会话冲突。

### 1.3 rounds 8–9:DFT-SP 标签永久丢失,要不要重算是决策题
- round8(~4,002 对)+ round9(~16,000 对)的 `dft_sp` 原始标签在 purge 中丢失,
  **无任何副本**(`cv_predictions.csv` 只覆盖到 round7)。见
  `memory/benzoin_dg_repo_moved_and_data_loss.md`。
- 现状:**rounds 1–7 是可信基线**(重建表重训 CV MAE 1.877 vs 历史 1.883,已验证,
  `memory/rounds17_reproduction_confirmed`)。round8/9 的 champion(blend MAE 2.106→2.074)
  依赖丢失的标签,目前**无法复现**,只剩 `gnn_norm_stats.joblib` + tabular joblib
  (能预测不能重训)。
- 重算成本:~20,000 个 DFT 单点(r2SCAN-3c/CPCM(DMSO),~0.64 核时/行)≈ **1.3 万核时**,
  墙钟视队列。几何(GFN2)可能还在 `benzoin_backups/cross_benzoin_xyz_archive/`
  (`cross_round8/9_xyz_geometry.tar.gz` 存在!)——若几何完整,只需重跑 SP,省掉构象搜索。
- **建议**:先不重算。rounds 1–7 champion 已足够作为当前生产模型;round8/9 的增量
  (MAE 1.877→~1.83 CV / 2.16→2.07 holdout)不足以justify 1.3 万核时,除非要发论文时
  需要完整 9 轮曲线。等 round10 组装完、看 round10 是否带来实质提升再定。

### 1.4 便宜的分析型待办(无需大算力)
- ✅ **homo dG 难尾巴 = g-xTB 基线失败**(已查,2026-09-02):150 个最差 P/磺酰基/亚胺/
  酰胺分子上 |g-xTB 基线误差| mean|·| 13.9 kcal,corr(残差, 基线误差)=0.888。是基线
  方法在高价 P/S 上崩,不是模型/标签问题。→ 推理时按子结构强制 route_to_dft。
  详见 `pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md`。
- ✅ **homo 主动学习不会提升**(已查):pool-AL 不适用(库已全标注);难尾巴 Boltzmann
  重标是 2× 确认的 null(冻结 MAE 10.01→10.13,boltz_corr ⟂ 残差)。详见
  `memory/homo-active-relabel-null-result` + commit `7e1ebad`。
- ⏳ **未做:推理时的 substructure→route_to_dft veto 规则**(P=O / 鏻盐 / 多磺酰基 / 多酰胺)。
  当前路由器只按 ensemble std 挑 top 15%,基线失败分子可能"自信地错"漏网。零训练成本。
- **醛 formyl C–H BDE 物理上限收紧**:`qc.py` 现在 `phys_max=250`、MAD k=6,漏掉了
  g-xTB 失败产生的 100–106 kcal/mol 毒标签(见
  `pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md`)。给醛侧单独加
  ~105–110 上限能去掉一小批。**不要直接改共用的 `qc.py`**——要单独前后对照评估
  (会影响历史排名数字可比性)。标为独立待办。
- **`products_all.csv` 会缺 `dG_gxtb_kcal` 列**:`featurize_product.py` 输出的是
  `dG_xtb_kcal`(ALPB),不是 g-xTB。B6 BDE 训练用不到 dG,但 `pipeline/analysis/`
  下的 `headroom_probe.py` / `adchqtaim_compare.py` 等会缺这列——需要时从恢复的
  `products_bdfe_gxtb_descriptors.csv` 或另跑 `gxtb_baseline.py` 回填。
- **nitro Δ_SP 双峰的 3 个失败分子**(id 86691 / 119366 / 121593):可细看它们的
  DFT 优化构象 / 硝基扭转角,确认是不是 g-xTB SCF 在多硝基上的已知 pathology。低优先级。
- **BDE 侧结果 `.pt` / JSON 的备份钩子**:重训完必须 `git add -f` + 加进
  `submit_backup_recovery_artifacts.sh`(purge 教训)。

---

## 2. 磁盘 / inode / 内存安全

`myquota`(2026-09-02 17:00):

| 文件系统 | 空间 | inode |
|---|---|---|
| **home4** (`schen3`) | 84 / 200 GiB(**42%**),软限 200 硬限 210 | ~400k / 1.0M(**40%**) |
| **scratch1** (`schen3@wstor_scratch1`) | ~1.3 / 10 TiB(**12.7%**) | ~1.95M / 4.0M(**48.8%**,软限 3.0M) |

- **空间:很安全**,两边都远未到限。
- **inode:scratch 是唯一要盯的**。已用 48.8%(接近软限 3.0M 的 65%)。
- **本次 BDE campaign 的 inode 代价被低估了**:每个 chunk 目录 ~200 个文件,**几乎全是
  `xyz/` 和 `ald_xyz/` 里的 `.xyz` 几何文件**(~100 pair × 2)。2209 chunk 全跑完
  ≈ **~44 万个文件** → scratch inode 会涨到 ~2.4M(软限 3.0M 的 ~80%)。
  **不会撞硬限,但要在 assemble 之后立刻清理**:
  ```bash
  find data/cross_benzoin/bde_homo_product_featurize_20260902 -type d \( -name xyz -o -name ald_xyz \) -exec rm -rf {} +
  ```
  这能立刻回收 ~40 万 inode。`.xyz` 对 B6 训练无用(用 SMILES + features.csv),
  且 gitignore 已排除 `*.xyz`。
- **教训**:以后跑 `featurize_product.py` 大 campaign,除非确实要留几何,应关掉 xyz
  输出(或跑完立即清)。round10 的 `conf_funnel` inode 效率修复
  (`HANDOFF_round10_20260721_ZH.md` §4.0)是同一类问题。
- **内存**:每个 array task 请求 48 CPU / 48 GB(genoa 最小计费单位),充裕,无风险。
- **quota 命令**:`quota -s` / `lfs quota` 在这台机器上会 hang,用 `/usr/hpc/bin/myquota`。

---

## 3. round10 之后要不要继续挖 round11?

**证据(round1–9 learning curve,`learning_curve_check_ensemble.py`,生产 MLP+XGB 架构,
5 折×5 重复 group-CV):**

| 数据比例 | n_pairs | CV MAE | R² |
|---:|---:|---:|---:|
| 0.25 | 7,020 | 1.842 | 0.765 |
| 0.50 | 14,040 | 1.797 | 0.781 |
| 0.75 | 21,061 | **1.770** | 0.796 |
| 1.00 | 28,081 | 1.773 | 0.655(R²/RMSE 恶化是重尾离群点采样伪影,非新不稳定) |

- MAE 从 0.25→0.75 单调下降,**还没到平台期**,但**边际收益在收窄**:
  第一次翻倍 −0.045,第二次 −0.027。
- round10(+16,000 对,总量 ~44k)≈ 再从 0.75 翻约一倍。按收窄趋势外推,预期
  CV MAE 再降 **~0.01–0.03**(holdout MAE 大概 2.07→2.05 量级)。
- **round11 的预期增量更小**(<0.01–0.02),基本进入"每轮一整轮 DFT-SP 换 <1% 相对
  提升"的区间。

**建议**:
- **round10 值得跑完**(已在跑,并行会话 job `26320177 score_r10` 正在做 AL 打分),
  把 learning curve 补到第 4 个点,给"是否继续"一个有把握的判断,而不是靠外推。
- **round11 默认不做**,除非:(a) round10 的实测点明显偏离收窄趋势(还在快速下降);
  或 (b) 目标从"降平均 MAE"转向"补特定弱项"。

**如果做 round11,用 hard-set 的证据做 targeted 采样(不是再抽一轮类别均衡大批)**:
现有 AL 循环 = 6 类均衡 + 含磷 stratum(n=800)+ ensemble-std 打分。hard-set
(`data/analysis/hard_set/`,8,150 行)显示 **cross-dG 的最差残差集中在**:
`flexible`(2,692)> `fg:amide`(871)> **`gxtb_baseline_failure`(667)** > `fg:sulfonyl`
(344)> `aliphatic`(284)> `fg:N_oxide`/`fg:nitro`/`fg:imine` > `fg:phosphorus`(20,
已被 sampler 过采样所以在 hard-set 里反而少)。

- 加 **amide-heavy / high-flexibility strata**(目前只有磷被单列)。
- 加一个 **`gxtb_baseline_failure` 采集项**:`score_round_active_learning.py` 现在只按
  epistemic uncertainty(ensemble std)排;基线失败的分子 ensemble std 可能很低
  (全体成员看同一个坏基线,自信地错)。应额外按预测的 |dG_gxtb − dG_orca|(在已标注
  数据上回归,或直接 SMARTS 标 P=O/鏻盐/多磺酰基)排一路,并入选池。
- **这些基线失败分子在推理时要强制 route_to_dft**(见 §1.4 和
  `pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md`)——对它们,
  再多训练数据也只是让 ML 修正学一个大 offset,治标;g-xTB 基线本身在高价 P/S 上不可靠
  是根因。
- 另一个比 round11 更值得的方向:**把 round8/9 标签补回来**(§1.3),先把已有数据用满,
  再谈新数据。

---

## 4. 各任务分别用哪个模型 / 表现如何

所有 BDE 数字为 **scaffold-disjoint** 划分(诚实泛化,不是 molecule-cold-split);
所有 cross-dG 数字为 **round1–9 的 n=450 scaffold-disjoint 冻结 holdout**。

### 4.1 homo 醛 formyl C–H BDE(g-xTB 标签)
| 模型 | MAE | R² | 备注 |
|---|---:|---:|---|
| **B6 = D-MPNN + H-SPOC 局部 3D 描述符(`x_d` 融合)** | **1.579** | **0.843** | 冠军;比 B4/B5 低 ~32% MAE |
| B4 D-MPNN(纯 2D 图) | ~1.60 | ~0.83 | 看不到局部电子结构 |
| B5 BonDNet 式(反应差分图嵌入) | ~1.92 | ~0.80 | 与 B4 互有胜负 |
| H-SPOC(局部 3D 描述符 + 调参 XGB) | 2.42 | 0.758 | **零新增计算**,性价比最高的基线 |
| D-SPOC-217 / B2 ECFP+XGB | ~R² 0.63 / 0.56 | | 传统基线 |
| B0/B1 ALFABET(zero-shot / 微调) | 无区分力 | | 域外物质 |

### 4.2 homo 产物中心 ketC–carbC BDE(g-xTB 标签)
| 模型 | MAE | R² |
|---|---:|---:|
| **B6(同上架构)** | **3.060** | **0.886** |
| B4 D-MPNN | ~3.64 | ~0.83 |
| B5 BonDNet 式 | ~3.69 | ~0.85 |
| H-SPOC 调参 | 4.05 | 0.818 |
| D-SPOC-217 | 5.365 | 0.715 |
| B2 ECFP+XGB | 6.18 | 0.652 |

→ **两个 BDE 任务的推荐都是 B6**。唯一稳健确认的架构结论是 `x_d` 融合(图嵌入 +
局部 QM 描述符)相对纯图 / 纯描述符的 35–47 个百分点优势;attentive pooling、
chemprop 2.3 MAB 等细粒度改动在 scaffold-disjoint 下不再成立(是泄漏驱动的假象)。
预算紧时 **H-SPOC + 调参 XGB** 是最好的"零新增计算"退路。

### 4.3 cross-benzoin 反应 ΔG(Δ-学习:`dG_pred = dG_gxtb + ML 修正`)
| 模型(round1–9,n=450 scaffold-disjoint holdout) | MAE | R² |
|---|---:|---:|
| g-xTB 基线(无 ML) | 3.905 | — |
| 单 XGB(260 特征) | 2.435 | 0.736 |
| MLP + 2×XGB ensemble | 2.163 | 0.786 |
| Attentive-pooling triple-GNN(GINE h128 l4) | 2.162 | — |
| **最佳 blend = 0.55·GNN + 0.45·ensemble** | **2.074** | — |

- bootstrap 确认 blend 优于 ensemble(P=0.988,B=20,000),和 round8 结论一致(P=0.9923)。
- **注意 scale 依赖**:round6 规模时 MLP+XGB ensemble 最好、GNN blend **没帮助**;
  round8/9 规模 GNN blend 的优势才稳定复现。→ 数据 <~8 轮时用 ensemble,≥8 轮上 blend。
- 历史 CV(5 折×10,泄漏偏乐观)口径:rounds1–7 ensemble MAE **1.877**(已复现)。
- 生产入口:`cross_benzoin/predict_cross_champion.py` 的 `CrossBenzoinBlendPredictor`。
- 根 `STATUS.md` 里 package 的 `predict_dG()`(63 特征旧 XGB,CV MAE 2.0)和
  `--champion`(72 特征 ENSEMBLE72,homo 全库 test MAE 1.503)是**打包好的稳定路径**,
  但不是上面这条 research 前沿。

### 4.4 homo BDE 模型 → 真实 cross 数据的迁移(给 cross-dG 供 BDE 特征时)
| 方案(round1–7 规模,18,025 cross 训练行,scaffold-disjoint) | MAE | R² |
|---|---:|---:|
| zero-shot(仅 homo) | 6.362 | 0.537 |
| homo 预训练 + cross 微调 | 4.501 | 0.695 |
| 仅 cross 训练 | 4.524 | 0.703 |

→ cross 数据 ≥18k 行后,**仅 cross 训练 ≈ 微调**,两阶段流程可省(round8+)。
数据稀缺的子类(某官能团)可能仍值得微调。

### 4.5 所有任务共同的最难子类
aliphatic > aromatic;重杂原子 + 强 EWG(Se 3.7×、nitro 3.0×、N-oxide 3.0×、
P 2.5×、imine 1.6×;产物侧还有 triflate、sulfonyl、amide)。**含磷是 cross-dG 的
单一最大误差驱动**(MAE 4.14 ≈ 基线 2 倍)。部分残差疑似是 g-xTB 标签噪声下限
(~2.1–2.7 kcal/mol 单构象)而非模型容量。

**2026-09-02 更新**:homo dG 那部分残差已定性为 **g-xTB 基线失败**(不是标签噪声、
不是模型容量)——见 §1.4。用 **hard-set dashboard**(`dashboard/`,数据
`data/analysis/hard_set/hard_set.parquet`,8,150 行)按 cause 浏览全项目最难分子,
每次重训后 `python pipeline/analysis/build_hard_set.py` 重建、`streamlit run
dashboard/hard_set_app.py` 查看,Progress 页看哪些难点被克服。当前 cause 分布:
`unexplained` 4,895 > `flexible` 2,692 > `fg:amide` 871 > `gxtb_baseline_failure` 667
> `fg:sulfonyl` 344 > `high_uncertainty` 329 > `aliphatic` 284。

---

## 5. 需要长期记录 / 已同步的事项

本文件 + 以下均已 commit 到 `agent/recovery-20260902`,并 rsync 到
`/home/schen3/benzoin_backups/docs_snapshot_20260902/`:

- `dashboard/` —— hard-set Streamlit 看板 + `pipeline/analysis/build_hard_set.py`
- `pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md`
- `pipeline/bde/STATUS.md` —— BDE 子课题权威入口
- `pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md`
- `RECOVERY_REPORT_20260902.md` —— purge 恢复全过程
- `cross_benzoin/docs/HANDOFF_round10_20260721_ZH.md` —— round10 配方 + learning curve
- memory 文件(`~/.claude/.../memory/`):`homo-active-relabel-null-result`、
  `bde-post-purge-asset-state`、
  `rounds17-reproduction-confirmed`、`benzoin-dg-repo-moved-and-data-loss`、
  `concurrent-cross-bde-session`(已更新为"单一项目"框架)、

- `pipeline/bde/STATUS.md` —— BDE 子课题权威入口
- `pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md`
- `RECOVERY_REPORT_20260902.md` —— purge 恢复全过程
- `cross_benzoin/docs/HANDOFF_round10_20260721_ZH.md` —— round10 配方 + learning curve
- memory 文件(`~/.claude/.../memory/`):`bde-post-purge-asset-state`、
  `rounds17-reproduction-confirmed`、`benzoin-dg-repo-moved-and-data-loss`、
  `concurrent-cross-bde-session`(已更新为"单一项目"框架)、
  `home-backup-archives-recovery`、`purged-binaries-and-envs`、`scratch-disk-quota-risk`

**尚未 push**(4 个 commit 在本地分支)。待用户确认后 push,或在并行会话的 round10
工作也落定后一起 push,避免分支竞态。
