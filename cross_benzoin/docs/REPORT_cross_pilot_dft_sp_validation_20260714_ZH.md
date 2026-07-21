# 交叉苯偶姻(cross-benzoin)试点：g-xTB 基线的 DFT-SP 验证(2026-07-14)

配套文件：`REPORT_cross_pilot_dft_sp_validation_20260714_EN.md`(English version)。

## 这个作业是什么

SLURM 作业 **24609263**(`pipeline/slurm/submit_dft_sp_cross.sh`，genoa 分区，
48 核，运行 2 小时 48 分钟)对 **598 行交叉苯偶姻试点数据**
(`cross_pilot_v1`，300 个无序对 / 598 个有向的 donor≠acceptor 组合，
在 aromatic_carbo / aromatic_hetero / aliphatic 三大类的全部 6 种组合中
分层抽样——详见 [[codex-cross-benzoin-session-20260714]])的产物几何结构
运行了 r2SCAN-3c 单点 DFT 计算(`pipeline/compute/dft_sp_cross_from_geom.py`)。
反应物(醛)一侧**完全不需要新计算**：每一对的两个反应物都已在
219k 行的同源(homo)campaign 中算过 r2SCAN-3c。只有这 598 个全新的
产物几何结构需要重新做单点计算。

这正是 `REPORT_codex_gap_analysis_20260714_EN.md` 中标记的首要待办验证关卡：
"一旦试点通过验证(ΔG 分布非平凡、AB≠BA 区域化学正确、错误率低)，
就扩大到 NEXT_STEPS.md Phase 2 要求的、由多样性/不确定性驱动的
几千行规模。"现在这三条标准都已用真实 DFT(而非仅 g-xTB)核实过。

## 结果一：零失败

598/598 行全部算出结果，**零错误**(`error` 列全程为空)。没有超时，
没有 SCF 不收敛，没有几何结构被拒绝。这是本项目迄今为止 DFT-SP
系列作业中完成率最干净的一次(同源 funnel_v3 campaign 在同量级
3600 秒超时下有约 4% 的 ORCA 超时率——见 [[dftsp-timeout-3600-snapshot]]；
但本次用了更长的 7200 秒超时且批量小得多，两者不能直接类比)。
不过对交叉产物几何结构本身而言，这是个好兆头：两个不同的
donor/acceptor 骨架拼在同一个产物里，并没有带来新的失败模式。

## 结果二：ΔG 分布非平凡、数值合理

`dG_orca_kcal`(真实 DFT，r2SCAN-3c/CPCM-DMSO)：均值 5.06，标准差 5.03，
范围 [−10.76, +24.29] kcal/mol，n=598。这是一个真实的分布，不是退化的
尖峰——形状上与同源苯偶姻的 DFT 标签分布相当，且整体偏正(放热性弱于
同源体系)，这在化学上是合理的：交叉配对混合了 donor/acceptor 的电子性质，
不像匹配良好的同源对那样为苯偶姻偶联"优化"过。

## 结果三：AB≠BA 区域化学在 DFT 层面得到确认

全部 299 个无序对都同时存在两个方向的行(598 = 299×2，有一对因醛缓存
缺失被丢弃)。上一次会话只在 g-xTB 层面确认了方向敏感性(300 对上均值
Δ 2.64 kcal)；这次作业在**真实 DFT 能量**上重复了这项检查：

| 统计量 | \|ΔG(A→B) − ΔG(B→A)\|，DFT(n=299 对) |
|---|---|
| 均值 | 3.70 kcal/mol |
| 中位数 | 3.07 kcal/mol |
| 最大值 | 15.47 kcal/mol |
| 最小值 | 0.02 kcal/mol |

在参考理论水平上，donor/acceptor 方向对 ΔG 是一阶效应，不是 g-xTB 的
伪影。这直接支持 `NEXT_STEPS.md` Phase 3/5 中采用"有向"而非
"对称配对"的建模架构。

## 结果四：g-xTB 基线能泛化到交叉产物，且失败模式符合预期

这是最核心的定量结果。将两个基线与真实 DFT 对比：

| 模型 | MAE (kcal/mol) | RMSE | 偏差(signed bias) |
|---|---|---|---|
| 原始 GFN2-xTB(`dG_xtb_kcal`) | 15.68 | 16.05 | −15.68(巨大的系统性低估) |
| **g-xTB(`dG_gxtb_kcal`)** | **3.45** | 4.57 | −2.33 |

正如预期，原始 GFN2-xTB 在绝对值上不可用(这与同源苯偶姻体系已确立的
结论一致：GFN2 单独使用需要 DFT/g-xTB 校正层——见
[[gxtb-dft-correction-champion]])。而 g-xTB——**没有任何针对交叉体系的
调优、没有 BDE 描述符、没有 ML 校正层，且在任何标定过程中从未见过
donor≠acceptor 的配对**——达到了 MAE 3.45 kcal/mol。作为参照：这
接近(略高于)本项目已确立的**同源 Δ-模型 ~3.2 kcal/mol 噪声下限**
([[delta-mae-noise-floor]])，高于调优后的同源冠军模型(MAE 1.503，
GNN 堆叠后 1.427——见 [[gxtb-dft-correction-champion]]、
[[gnn-stacking-confirmed-full-scale]])。换句话说：**g-xTB 零样本泛化到
一个化学上全新的领域(交叉产物)，相对于完全调优的同源模型大约多付出
2 kcal/mol 的 MAE 代价，但并未失效。**这是个好结果——说明现有的
g-xTB 单点计算，以及同源 Δ-模型在特征/架构上的选择，是交叉体系微调的
一个稳固起点，而不需要从零重建。

按类别对拆分的 g-xTB MAE(kcal/mol)：

| 类别组合 | n | MAE | 中位 AE |
|---|---|---|---|
| aliph-carbo | 46 | 2.19 | 1.82 |
| aliph-hetero | 16 | 3.12 | 2.41 |
| carbo-hetero | 250 | 3.18 | 2.32 |
| carbo-carbo | 128 | 3.41 | 2.49 |
| aliph-aliph | 4 | 4.28 | 4.50 |
| hetero-hetero | 154 | 4.28 | 3.44 |

这个排序(脂肪族参与的配对最容易，hetero-hetero 最难)与**项目中已确立的
同源苯偶姻失败模式一致**：吸电子基/富杂原子底物正是 GFN2/g-xTB
电子结构近似最容易失效的地方([[nonewg-outlier-drivers]]、
[[screen-v6-funcgroup-analysis]])。误差最大的 8 个异常值(|误差|
12.3–15.8 kcal/mol)全部是 carbo-hetero、carbo-carbo 或 hetero-hetero
配对，没有一个是脂肪族的。这是**同一个**已知失败模式在新的(交叉)
化学空间中再次出现，而不是新的失败模式——这对于同源数据上已经有效的
校正策略(BDE/BDFE 描述符、EWG 感知特征)能否迁移是个令人放心的信号。

P90/P95 绝对误差：7.66 / 9.77 kcal/mol——尾部适中，集中在上表中
富杂原子的类别里。

## 结论：试点验证关卡——通过

| 标准(来自上一份报告) | 结果 |
|---|---|
| ΔG 分布非平凡 | ✅ 均值 5.06，标准差 5.03，跨度约 35 kcal/mol |
| AB≠BA 区域化学正确 | ✅ 在真实 DFT 层面确认，方向效应中位数 3.07 kcal/mol |
| 错误率低 | ✅ 0/598(0%) |
| (附加)g-xTB 基线在调优前即可用 | ✅ MAE 3.45，比调优后的同源冠军高约 2 kcal，失败模式与同源体系相同 |

这个试点被设计用来检验的每一条关卡都已通过。项目现在具备条件进入
**Phase 3(组装交叉训练表并训练首个 tabular Δ-模型 / g-xTB 基线)**
(依据 `NEXT_STEPS.md`)，可以直接用这 598 行作为首批标注数据，
也可以先再跑一轮多样性/不确定性驱动的扩充再训练——见下方尚未
解决的决策点。

## 尚未解决的决策(本次会话未定案)

`NEXT_STEPS.md` / `CROSS_BENZOIN_ML_RECOMMENDATIONS.md` 中的主动学习循环
明确建议：先在一个适度规模的多样化子集上训练首个集成模型，再针对
高不确定性区域挑选*下一轮*标注数据，而不是在从未训练过模型之前就
盲目扩大标注集规模。598 行足够拟合一个初步的 g-xTB 校正基线并获得
校准过的不确定性估计，但对于可靠的"分子不相交"留出测试集而言可能
还是太小。目前有两条合理路径，尚未选定：

1. **现在就训练**：用这 598 行组装 Phase 3 的交叉训练表(按
   `DESCRIPTOR_POLICY_CROSS.md` 构建 donor/acceptor/product/delta
   角色感知描述符)，拟合首个 g-xTB 校正 Δ-模型，获得逐行不确定性，
   并用它来挑选下一轮主动学习的批次。
2. **先扩充规模**：在投入下一轮 DFT 预算之前，再跑一轮多样性分层的
   试点(几千行规模，仅 xTB/g-xTB，不做 DFT)，这样在切出
   训练/验证/测试不相交的划分时能有更多行可用。

## 补充：本次会话已做出决定并训练出首个交叉 Δ-模型

在获得自主推进的授权后，选择了路径一("现在就训练")：598 行足够拟合
一个首个校准基线，而且这比盲目扩大标注集更贴合 `NEXT_STEPS.md` 自己
定义的主动学习循环。两个新脚本实现了 Phase 3 的第 1–2 步：

- `cross_benzoin/assemble_cross_training_table.py`——按
  `DESCRIPTOR_POLICY_CROSS.md` 组装角色感知训练表：`donor_*`/`acceptor_*`
  (醛描述符按**规范化 SMILES** 而非 `id` 列拼接——`aldehydes_all.csv` 的
  `id` 其实是 `aldehydes_clean_v6.csv` 里的行号，和产物表中
  `donor_id`/`acceptor_id` 使用的 InChIKey 是两套不同的键；第一次运行时
  踩到了这个坑，在任何模型看到数据之前就已修复)、`product_*`(来自
  `cb_featurize.py`，已是角色感知的)、三个角色各自的 RDKit-2D 描述符，
  以及一个 `interaction_*` 区块(HOMO(D)−LUMO(A) 能隙、Fukui 匹配度、
  空间/电子失配项)。ADCH/QTAIM 列全部为空(这个试点从未跑过
  Multiwfn)，直接丢弃而非插补。输出：
  `data/cross_benzoin/cross_pilot_v1/cross_train_table.parquet`，
  598 行 × 147 个特征，299 对。
- `cross_benzoin/train_cross_delta.py`——用 XGBoost(较浅：深度 3，
  300 棵树，照顾到 n=598)预测校正量 `dG_orca_kcal − dG_gxtb_kcal`。
  交叉验证按**无序对键(pair_key)做 GroupKFold**，而非普通 K-fold——
  AB/BA 共享两个母体分子，不分组的 CV 会让分子身份信息在训练/测试间
  泄漏。为压低这个规模下的划分噪声，重复 5×20 次(与同源模型重复
  K-fold 的理由相同)。同时跑了 `DESCRIPTOR_POLICY_CROSS.md` 要求的
  完整消融实验。

**结果：MAE 2.09(RMSE 2.70，R² 0.71)，相对 g-xTB 基线 MAE 3.44**——
一个没有 BDE/BDFE 描述符、没有 GNN、只用 598 行训练数据(比 22 万行的
同源 campaign 小两个数量级)的首版模型就取得了 **+1.35 kcal/mol** 的
提升。这还没达到调优后的同源冠军模型水平(1.503/1.427)，考虑到
数据规模的差距这是预期之中的，但它证实了交叉 Δ-学习这条路径本身是
可行的，而且——和 g-xTB 基线本身一样——恰好在 g-xTB 最弱的地方
校正力度最大：hetero-hetero 的 MAE 从 4.28 降到 2.39，carbo-carbo 从
3.41 降到 1.79，而 aliph-carbo(g-xTB 本来就最容易的类别)只从
2.19 降到 1.73。

消融实验结果(特征区块，同一套 CV 协议)：

| 区块 | 特征数 | MAE | RMSE | R² |
|---|---|---|---|---|
| 仅 2D | 48 | 2.91 | 3.71 | 0.454 |
| 仅醛(donor+acceptor 原始特征) | 52 | 2.70 | 3.49 | 0.517 |
| **仅产物** | 53 | **2.24** | 2.89 | 0.669 |
| donor+acceptor(原始+2D) | 100 | 2.72 | 3.48 | 0.520 |
| 全部原始区块(不含交互项) | 137 | 2.08 | 2.68 | 0.716 |
| 全部+交互项 | 147 | 2.09 | 2.70 | 0.711 |

有两点值得为下一轮标注留意：(1) **仅产物侧描述符就已经拿到了大部分
信号**(MAE 2.24 对比完整模型的 2.08–2.09)——与同源模型"反应中心的
低层物理量携带了大部分信息"这一结论一致，现在在交叉体系上也得到了
确认；(2) **手工构建的交互项区块在当前规模下没有帮助**(2.094 对比
不含交互项的 2.077)——这是个小幅倒退，更可能是 598 行上 10 个额外
相关特征带来的噪声/轻微过拟合，而不是真正的负信号，值得在标注集
扩大后重新检验。SHAP 显示 `P_int`(产物的全局空间/静电指标)是
遥遥领先的第一特征，其次是 acceptor 侧的 WBO/偶极矩/`P_int`——这在
化学上是合理的，因为 acceptor 一侧会变成亲电的碳醇中心。

产出文件：`data/cross_benzoin/cross_pilot_v1/train_v1/{models,figs,data}/`
(模型 joblib、特征列表、含完整消融表的元数据、parity 图、按类别的
残差图、SHAP 重要性图、CV 预测 CSV)。

**仍未解决**：这只是单一随机种子下的按对分组 CV，还不是文档 Phase 4
晋升关卡要求的、冻结不动的分子不相交测试集；这个 598 行模型是用来
指导下一轮主动学习标注的首次读数，而不是可以晋升的候选模型。按照
主动学习循环，下一步自然是：用这个模型逐行的 CV 残差/不确定性去
挑选下一轮标注——重点覆盖高误差区域(富杂原子配对)和当前代表性不足
的类别(aliph-aliph、aliph-hetero——这里分别只有 4 行和 16 行)。
