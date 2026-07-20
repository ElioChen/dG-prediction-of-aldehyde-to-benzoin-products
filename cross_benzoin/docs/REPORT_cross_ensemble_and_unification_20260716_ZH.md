# Cross-benzoin：MLP+XGB 集成打包、homo+cross 融合大规模复验、round 6

**日期**：2026-07-16
**承接**：`REPORT_cross_round3_active_learning_20260715_{EN,ZH}.md` 以及 round4/5 规模扩张工作（见 memory `cross-round4-5-scaleup-and-architecture-win-20260716`）。

> **2026-07-17新增勘误**：本报告中所有冻结holdout MAE数字，都是在`candidates_v3`最初那版
> 按分子级别（InChIKey不相交）划分的切分上测出的，后来发现该切分严重泄漏了Bemis–Murcko
> 骨架（29行留出集中93%的行至少有一侧骨架已经出现在训练集中）。请将本报告中的冻结holdout
> 数字视为偏乐观（相对真实的全新骨架泛化能力，高估约0.2–0.5 MAE）。修正后的骨架不相交
> 评估（n_test=450）以及推荐的生产模型（`cross_benzoin/predict_cross_champion.py`）现已
> 取代本报告的headline指标——详见`STATUS_ZH.md`的勘误说明以及memory
> `cross-scaffold-disjoint-rebuild-20260717`。本报告下文内容保持不变，作为历史快照保留。

## 背景

上一次会话（2026-07-15/16）确认了三件事，但都停留在不同的未完成阶段：

1. MLP+XGB 集成架构在交叉验证上优于单一 XGB 冠军模型，且其优势随数据规模**扩大**（4120 行时相对 MAE -4.9%，17270 行时 -9.5%）——但这只是 CV 层面的结果，既没有冻结的分子不相交 holdout 数字，也没有可交付的模型文件。
2. homo+cross 融合（在 cross 自身数据基础上加入 homo/自缩合库的 30,000 行分层抽样）在旧的 4120 行 cross 规模下确认了真实但适度的增益（仅 cross 行的 CV MAE：2.966 → 2.910，相对 -1.9%）——此后 cross 自身数据几乎翻两番（round4/5 → 17,270 行）且 mordred 描述符已并入流水线，但一直未重新测试。
3. round5 只用掉了 screen10k 原始候选池 10,177 对中的 2,500 对，剩下 7,677 对已完成特征化但一直闲置。

本次会话把这三个缺口全部补上。

## 一、MLP+XGB 集成：冻结 holdout + 交付模型

新脚本 `cross_benzoin/train_cross_ensemble.py`，沿用 `train_cross_delta.py` 的约定以保证两者可直接对比：

- **特征集修正**：此前的纯 CV 实验（`architecture_ensemble_experiment.py`）用了表里**所有**列，包括已经三次被证明无用的 10 个 `interaction_*` 交互项。新脚本改用冠军模型确切的 `all_raw_blocks+mordred` 特征选择（260 个特征，排除交互项）——这才是真正的同口径对比，而不只是"CV 流程相同、特征集不同"。
- **冻结 holdout**：复用与冠军模型完全相同的 `pair_split_labels()` / `frozen_holdout_eval()` 机制，在 `candidates_v3` 的 `train` 切分行上拟合，在同样冻结的 29 行 `test` 切分行上评估一次。
- **打包**：一个 `MLPXGBEnsemble` 类（StandardScaler + MLP(128,64) + XGB(depth=5) + XGB(depth=7)，三者均值），把插补、标准化、平均全部封装在单一的 `.predict(df)` 调用背后——用 `joblib.dump` 打包，方式与单 XGB 冠军模型完全一致，并配有匹配的 `feature_list.json` / `metadata.json`。在信任完整跑批结果之前，先验证了 `joblib.load` → `.predict()` 在留出行上的往返正确性。

**结果**（round1-5 表，17,270 行 / 8,642 对，5×10 重复分组 CV）：

| | 单 XGB 冠军模型 | MLP+XGB 集成 | Δ |
|---|---:|---:|---:|
| CV MAE | 2.397 | **2.176** | -9.2% |
| CV R² | 0.409 | 0.451 | |
| 冻结 holdout MAE（n=29） | 2.983 | **2.633** | -11.7% |
| 冻结 holdout R² | 0.865 | **0.898** | |

两个数字的变化方向和幅度都与上次会话仅凭 CV 得出的估计一致，证实了架构优势是真实的，不是 CV 层面的假象——它在真正的分子不相交 holdout 上同样成立。按反应类型分类看，集成模型绝对误差最大的是 `aliph-aliph`（2.69 kcal），最小的是 `hetero-hetero`（2.04 kcal），与本项目一贯的难度排序一致。

**交付产物**：`data/cross_benzoin/cross_round5/train_ensemble_slim120_v1/models/cross_ensemble_model.joblib`（附 `feature_list.json`、`metadata.json`、parity/residual/XGB 重要性图、`cv_predictions.csv`、`mae_by_category.csv`）。这是目前验证最充分的 cross-benzoin 模型，是成为新冠军模型 / 未来主动学习轮次新打分模型的自然候选，具体是否提升还待用户决定。

## 二、homo+cross 融合：17,270 行规模复验

新脚本 `cross_benzoin/assemble_cross_training_table_unified_v2.py` 把此前仅支持 round1-3（且无 mordred）的融合脚本推广到 round1-5，并加入**与原测试完全相同、未改动**的 30,000 行分层 homo 样本——这样能单独隔离出"cross 自身数据增长 + mordred"带来的影响,而不是 homo 端发生了变化。

**特征口径对齐修正**：原始融合表有 2,180 列特征（homo 的产物侧 mordred 文件是完整的 1,826 描述符全量导出，而 cross 自身的产物 mordred 文件只覆盖其中 219 列）。如果直接用全部 2,180 列训练，模型会依赖那些对每一行 cross 数据都结构性缺失（NaN）的 homo 专属列——这并不是"homo 是否帮助 cross"这个问题该衡量的东西。修正方法是训练前把融合表精确裁剪到冠军模型的 260 个具名特征；这 260 个特征全部按名称存在，且各 round 间 NaN 比例低且均匀（2.5%-7.6%），证明经过 SHAP 精简的 mordred 特征集在两个数据群体间都能良好泛化，而非专属于 cross 或 homo 任一方。

**结果**（47,270 行：17,270 cross + 30,000 homo，同样的 260 特征口径，同样的 5×10 CV 流程）：

- 冻结 holdout MAE：**2.983**，与纯 cross 跑批**逐字节完全相同**。这是预期行为，不是 bug——`frozen_holdout_eval()` 只对带有 `candidates_v3` 切分标签的行拟合，而每一条 homo 行的 `donor_id` 都是数字型醛库索引（不是 InChIKey），因此全部 30,000 行结构性地被排除在该拟合之外。冻结指标从设计上就看不到 homo 的影响。
- 混合 5×10 分组 CV 的 MAE：2.279（相比纯 cross 的 2.397）——但这个平均值被 homo 那 30,000 行（占 63%、整体更"容易"）主导，本身并不能直接回答 cross 预测是否真的变好了。
- **真正能回答问题的数字**——对混合跑批的 `cv_predictions.csv` 做按 round 拆分（通过 'id' 合并回原表的 'round' 列），只看 17,270 行 cross 数据部分：**MAE 2.374**，对比纯 cross 模型自身的 CV MAE **2.397**。

| 规模 | 纯 cross CV MAE | 融合模型（仅看 cross 行）CV MAE | Δ | 相对变化 |
|---|---:|---:|---:|---:|
| round1-3（4,120 行），上次会话 | 2.966 | 2.910 | -0.056 | -1.9% |
| round1-5（17,270 行），本次会话 | 2.397 | 2.374 | -0.023 | **-1.0%** |

**解读**：homo+cross 融合在新规模下依然叠加有效——效应方向干净地复现了——但**幅度大约减半**，而 cross 自身数据同期增长了 4.2 倍。这与上次会话提出的假设（"随着 cross 自身数据增多，它对 homo 帮助的需求会降低"）相符,而非被推翻。这里的绝对增益（0.023 kcal/mol）已经小到接近换一种折叠划分时重复 CV 噪声本身就能产生的量级（融合跑批的折叠结构必然与纯 cross 跑批不同，因为总行数/分组数不同）——所以应该理解为"效应真实存在、且在两个独立规模上方向一致地有益，但已不再是一个能自信断言的大杠杆",而不是一个精确、无噪声的 +0.023 kcal 数字。

## 三、Round 6：候选池重新打分、选出 4,000 对、DFT-SP 已提交

上次会话为这个目的专门建立的 10,177 对 screen10k 候选池（round5 只用掉其中 2,500 对）还剩 7,677 对（15,342 行）完全特征化但闲置。

- 把候选特征表过滤到未被选中的 7,677 对（`cross_round6_candidates.parquet`）。
- 用 `score_round_active_learning.py`（脚本本身未改动）针对**新的 round1-5 slim120 冠军模型**重新打分（40 次自举、按分子对分组的不确定性集成），而不是继续用当初给这批候选池排序时那个已过时的 round1-3 模型——让这个信息量已大幅提升的新模型重新审视自己还剩哪些盲区。
- 按不确定性选出**前 4,000/7,677 对**——是 round5 批量的 1.6 倍，延续用户要求的"批量继续放大"模式，同时特意只用掉约一半候选池，为未来一轮留出 3,677 对备用，而不是一次性耗尽这个储备池。
- 构建双方向 DFT-SP 输入（7,996 行；4,000 对中有 4 对只有一个方向在特征化阶段存活下来）。清单检查：7,996 行中 7,906 行（98.9%）具备完整缓存的醛端 DFT 能量，与此前各轮的覆盖率一致。
- 在提交完整数组作业前，直接对 `dft_sp_cross_from_geom.py` 做了冒烟测试（2 个真实 ORCA 单点计算顺利完成，ΔG 数值物理上合理）。
- 以 SLURM 数组作业 **24667830** 提交（80 个 chunk × 每 chunk 100 行，genoa 分区，%50 节流——这个节流比例已在上次会话中针对这个具体脚本验证过是安全的，因为 DFT-SP "一个任务对应一个 ORCA 进程"的资源特征不会重现 `cb_featurize` 那种节点本地临时空间超订的失败模式）。**截至本报告撰写时已完成 50/80 个 chunk**，仍在运行中。

**尚未完成的部分**（供后续接手者参考）：(1) 等待 round6 数组作业跑完；(2) 执行 round4/5 都需要的同一个产物 BDE 补算步骤（`calc_bde_gxtb_product_cross.py`）；(3) 组装 round1-6 合并表，在其上重新训练单 XGB 冠军模型和新的 MLP+XGB 集成模型；(4) 在 round1-6 规模上第三次重跑 homo+cross 融合测试，看这个"增益递减"的趋势是继续、走平还是反转；(5) 决定是否把 MLP+XGB 集成提升为 `score_round_active_learning.py` 用于未来轮次不确定性打分的"在役冠军模型"。

## 结论

- cross 目前验证最充分的最好数字，是 **MLP+XGB 集成的冻结 holdout MAE 2.633（R² 0.898）**——相对单 XGB 冠军模型（2.983 / 0.865）的真实、经 holdout 确认的提升，不只是 CV 层面的假象。
- homo+cross 融合在数据规模扩大 4 倍后依然确认有效，但增益已明显收窄（-1.9% → -1.0%），与"cross 自身数据越多、对 homo 帮助的依赖越小"这一假设相符——这是判断是否值得继续优先投入融合方向、还是应该优先跑更多主动学习轮次的重要参考数据点。
- round6 已在跑,且没有产生任何新的 xTB/g-xTB 计算成本（纯粹复用已有储备池），继续朝着 homo 1.4-1.5 MAE 这个上限缩小差距。
