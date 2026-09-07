# benzoin-dg 项目完成总结（09-04 首版，**09-07 增补更新**，供工作汇报用）

> 事无巨细、按时间顺序的项目总览，以 **cross-benzoin ΔG 主动学习（AL）** 为主线，
> 附 BDE 预测子课题、homo dG 项目、以及贯穿全程的两次基础设施事故（git 损坏、
> scratch 全量 purge）。配套一份可分享的 Artifact 报告页（精简版，09-04 版本，
> 09-07 的增补尚未同步进去）。
>
> **09-07 更新说明**：按项目惯例本文件本应是一次性快照（见文末），但应用户要求
> 用作工作汇报底稿，做了一次增补编辑而非另开新文件——新增内容集中在 §3.9（几何
> 消融最终判决）、§3.10（Goal 3 从"验证"到"已交付为工具"）、§4.4-4.5（BDE 全量
> 重训出新冠军 + 一次跨子项目 bug 的发现与修复）、§7（09-07 状态快照）。09-07 之
> 后的进展仍然去看 `RUN_LOG_20260903.md`（living doc）。
>
> 权威 living 文档索引见文末「文档地图」。

---

## 0. 一句话 + 关键数字

**目标**：给任意一对醛（donor + acceptor，可同分子）预测苯偶姻缩合反应的 ΔG
（kcal/mol），用主动学习挑选最有信息量的下一批 DFT 计算，把"哪些反应对值得
做 DFT"这件事从盲选变成模型驱动的排序问题。

| 里程碑 | 数字 |
|---|---|
| AL 轮次 | 10 轮完整闭环（round1→round10），2026-07-14 启动，2026-09-04 round10 落地 |
| 当前 champion | **r1-10 blend**（MLP+XGB ensemble + attentive-pooling GNN，w_gnn=0.5） |
| 当前 champion 诚实 MAE（骨架不相交 holdout, n=448） | **2.215 kcal/mol**（g-xTB 物理基线 5.037，相对基线 −56%） |
| 训练数据规模 | 35,528 行 / clean-train 22,771 行（10 轮 AL 累计） |
| **标签质量天花板（09-04 新发现）** | 单构象 r2SCAN-3c DFT 标签噪声 std ≈ **2.9 kcal/mol** —— champion MAE 已贴地板 |
| **round10 主动学习消融（09-04 新发现）** | AL 诊断出真实盲点，但训练收益 **≈0（−0.03，噪声内）** |
| **重新表述为分类/排序（09-04 验证，09-07 已交付）** | 同一个 champion 做"favorable/unfavorable"判别 **AUC 0.92-0.93**，top-10% 精度 **64%**（g-xTB 基线仅 31%，随机 10%）—— **不受标签噪声天花板限制**；09-07 起已作为 `predict_dg.py` 的标准输出列（非单独 eval 脚本）|
| **几何方法偏差消融（09-04 收官）** | 60 对三物种 ΔΔG 消融：hetero 组中位 ddG −0.47 kcal vs control −0.25 kcal，均远低于 1 kcal 判定阈值 → **无显著偏差，标签质量调查结案，不做定向重标签** |
| 姊妹子课题：BDE 预测（**09-06 全量重训，新数字**） | champion 现为 **B6 5-seed deep ensemble**，骨架不相交全量 220k：醛 MAE **1.851** / 产物 MAE **2.826**（旧数字 1.579/3.060 是 42k 局部库上跑的，已作废） |
| **跨子项目 bug（09-06 引入，09-07 发现+修复）** | BDE 库重建意外让 `predict_dg.py` 对任何新分子对 100% 失败（共享库文件的覆盖率哨兵字段选错）；已修复+部署，见 §4.5 |
| 姊妹子课题：homo dG（A+A，已上线） | 219,364 个 DFT 标签全覆盖，champion 测试 MAE 1.503 |
| 基础设施事故 | 2026-07-13 git 数据库损坏（历史重开）；2026-07-20~29 scratch 全量 purge（大量数据/权重丢失，本文档记录完整恢复过程）；2026-09-03 深夜 scratch inode 硬顶 133% |

---

## 1. 项目结构：三条线 + 关系

```
benzoin-dg（同一个 repo，同一个化学问题的三个角度）
├── homo dG      —— A+A 同源苯偶姻缩合，Δ-learning，已上线（本总结从略，见 §5）
├── cross-benzoin ΔG —— A+B 交叉缩合，Δ-learning，10 轮主动学习闭环（★ 主线，§3）
└── BDE 预测     —— 直接预测反应中间体的键解离能，供 cross 项目消费其 g-xTB BDE
                     特征、也是独立预测子课题（§4）
```

三条线共享：醛库（220,859 个同源醛，g-xTB 全标注）、g-xTB/DFT 计算基础设施、
`pipeline/compute/` 通用几何+能量代码、同一个 SLURM 账号和 scratch 配额。
2026-09-02 起被 2026-07 purge 逼着做了一次三线联合恢复，此后按"窗口 A（BDE）/
窗口 B（cross dG）"分工推进，2026-09-04 起合并为单会话统一驱动（本文档即产出于
合并后的会话）。

---

## 2. 时间线总览（跳转用；细节见对应小节）

| 日期 | 事件 |
|---|---|
| （早于 07-13） | homo dG 项目已完成、已上线（champion ENSEMBLE72，测试 MAE 1.503） |
| **2026-07-13** | git 对象库损坏，历史重开（"Fresh history"）——早于此的提交历史丢失，只留代码现状 |
| 2026-07-14 | **cross-benzoin v2/v3 启动**：4M 候选对数据集、featurize 流水线、pilot 600→4200 行验证 |
| 2026-07-20 | **BDE 预测子课题 Phase 1 完成**（B0-B6 七个基线跑完，B6 夺冠）；round1-8 主动学习闭环跑完；split 比例方法论定案（80/10/10） |
| **2026-07-17** | ⚠️ **骨架泄漏勘误**：发现历史每一个"冻结留出集 MAE"都在会泄漏骨架的切分上测出，重建真正骨架不相交切分，champion 数字全部重估（更诚实但更差） |
| 2026-07-21 | round9 完成（16,000 候选对）；round1-9 champion 确认（blend MAE 2.074）；round10 决策文档写就；3D-GNN 架构探索（side track，null 结果） |
| **2026-07-20~29（约）** | **Snellius scratch 全量 purge**：大量 checkpoint（`.pt`）、中间数据、round8/9 DFT 标签永久/暂时丢失 |
| 2026-09-02 | **购后恢复启动**：repo 迁至 `benzoin-dg-restored`，三线并行恢复（BDE 描述符库重算、cross rounds1-7 复现验证、环境重建） |
| 2026-09-03 | round8/9 DFT 标签**从零重算**（~41.5k 单点能）；cross GNN 全部重训；深夜 **scratch inode 硬顶 133%**（BDE 几何归档跟不上，账号级 EDQUOT） |
| 2026-09-04 凌晨 | 隔夜自主流水线跑通：r1-9 champion+GNN+blend 完成（**MAE 2.167**），round10 用 r1-9 重打分 |
| 2026-09-04 上午 | round10 三腿 DFT 完成（1969 对新标签）→ **r1-10 champion 确认**（**MAE 2.215**）；**round10 AL 消融 = null** |
| 2026-09-04 中午 | 标签质量三探针：构象噪声地板（~2.9 kcal）、GFN2 vs g-xTB 几何偏差（结构化于 P/S/B）、泛函偏差（探针本身失败，放弃） |
| 2026-09-04 下午 | 会话交接三次（09-03/09-04 各一份 HANDOFF）；决定性几何-方法消融 `dg_geom_method` 在跑；本会话**接管全项目**（含此前"别碰"的 BDE 自动化线）；**分类/排序重表述验证为正**；本总结文档产出 |

---

## 3. 主线：cross-benzoin ΔG 主动学习闭环（10 轮）

### 3.0 问题设定

- 反应：donor 醛 + acceptor 醛（可同分子）→ 苯偶姻类产物，`dG_pred = dG_gxtb 基线 + ML 修正量`（Δ-learning）。
- 标签：`dG_orca_kcal`，r2SCAN-3c / CPCM(DMSO) 单点能，`dG = G(prod) − G(donor) − G(acc)`。
- 特征：260 维冻结 schema（QM 描述符 + mordred + 反应互作特征），从 round7 起未变。
- 评估资源：骨架不相交（Bemis-Murcko scaffold-disjoint）held-out 集，2026-07-17 后固定在 **n=448~450**。
- 模型：单 XGB → MLP+XGB ensemble → +attentive-pooling GNN 的 50/50 blend（三代架构演进，见下）。
- 采样：round1 类别多样性；round2 起全部是 **bootstrap ensemble 不确定性主动学习**
  （用当前 champion 对候选池打分，按预测方差降序选下一批做 DFT）。

### 3.1 Round 1-6（2026-07-14~16，早期闭环，旧的会泄漏骨架的切分口径）

| | R1-3 | R1-4 | R1-5 | R1-6 |
|---|---:|---:|---:|---:|
| pairs / 有向行数 | 2,062 / 4,120 | 6,194 / 12,378 | 8,642 / 17,270 | 12,597 / 25,176 |
| CV champion MAE | 2.966 | 2.261 | 2.397 | 2.276 |
| 冻结集 champion MAE † | 3.398 | 3.043 | 2.983 | 3.132 |
| 冻结集 ensemble MAE † | — | — | 2.633 | 2.582（当时最佳） |

（† = 下述 07-17 勘误之前的旧切分数字，仅作历史记录，**不要引用为真实精度**。）

采样细节：round1 靠类别多样性铺开覆盖面；round2-6 全部改为 bootstrap 集成不确定性
AL；round6 先对 screen10k 候选池剩余部分重新打分，再按不确定性挑出前 4,000/7,677 对。
GNN（三编码器，同源预训练+交叉微调）在 round5 规模（n=29 冻结集）曾以 P=0.987 显著
优于 ensemble，round6 规模（更大但仍 n=29）**未复现**（P=0.456）——当时判定为超参数
未调，而非架构本身无效（后续证实：调好 lr/patience 后确实稳定优于 ensemble）。

**已知小缺口（早期即记录，长期未处理，不影响结论）**：`interaction_*` 10 个特征
组确认无用；醛库 ~0.15% DFT-SP 超时缺口；产物 BDE 计算成功率稳定 ~94%（部分环系
几何/成键判定失败）。

### 3.2 2026-07-17：骨架泄漏勘误（方法论转折点）

**发现**：`candidates_v3` 最初那版"分子级别（InChIKey 不相交）"切分，虽然分子不
重叠，却让 **93% 的旧冻结留出集（29 行）在骨架层面早已出现在训练集里**（只有
2/29 对是真正骨架全新的）。独立的骨架不相交 5 折重复 CV（n=2255）证实了一个
真实、可复现的 **+0.221 MAE（~9.8% 相对）泛化差距**，出现在"内插"与"面对全新
化学骨架"之间。

**结论**：本项目历史上报告的**每一个**"冻结留出集 MAE"都把真实泛化能力**高估
了约 0.2-0.5 MAE**。

**修复**：`rebuild_scaffold_disjoint_split.py` 对全部骨架做全局贪心装箱，重建真正
骨架不相交的 80/10/10 切分（干净训练集 19,687，**干净测试集扩大到 450**，比旧的
n=29 大约 15.5 倍、统计上可信得多）。round1-7 全部数据按新切分重新打标签，两条
架构线（ensemble、GNN）都在新切分上重训。

**诚实的新 headline（round1-7）**：单 XGB MAE 2.448/R² 0.730，MLP+XGB ensemble
MAE 2.256/R² 0.763，**blend MAE 2.215**（bootstrap P(blend>ensemble)=0.9632，90% CI
不跨零）。—— 注意这个 2.215 和当前 09-04 的 champion 数字**碰巧数值相同**，但
分别对应 round1-7（旧标签）和 round1-10（recovery 后重算标签），是两个不同的
估计，纯属巧合。

同一时期还定案了 **切分比例方法论**（2026-07-20）：实测 80/10/10 vs 70/20/10，
80/10/10 clean-train 更多（19,687 vs 14,874）、blend MAE 更低（2.215 vs 2.614）、
显著性检验更干净（90% CI 不跨零 vs 勉强跨零）——80/10/10 定为生产标准，70/20/10
留档对照，此后一直沿用。

### 3.3 Round 7-9（2026-07-17~21）：champion 架构演进 + 学习曲线证据

- **Round 7**：screen10k 候选池耗尽（第5轮2,500+第6轮4,000+第7轮3,677=10,177/10,177
  全部用完）。R1-7 达到 32,456 行/16,241 对，CV ensemble MAE 1.883（当时最佳，旧切分口径）。
- **Round 8**（2026-07-20，用户批准，利用 genoa 空闲算力启动）：从新骨架切分过滤后的
  1,263,002 对可用池里，按类别配平抽 4,002 对/8,004 行有向数据（round8 pairs）。
  执行中发现标准流程漏了一步（缺 53 个 `product_mordred_*` 特征，champion schema
  从 round6/7 起就需要）——修复为脚本层面自动探测 mordred 侧车文件，探测不到就硬报错
  退出，而不是悄悄产出残缺表。**架构升级**：引入 **attentive-pooling GNN**（3-seed
  确认稳健：GNN-only MAE 2.324 vs 默认架构 2.563）。
  R1-8：clean-train 27,583，ensemble-only 2.201，**blend 2.106**（bootstrap 90% CI
  (0.031, 0.159)，P=**0.9923**）——比 round1-7 显著更好，attentive 架构确认有效。
- **Round 9**（2026-07-21）：16,000 候选对特征化+BDE+mordred+AL打分+DFT-SP 全部落地。
  R1-9：clean-train 43,367，ensemble-only 2.163，GNN-only 2.162，**blend 2.074**
  （w=0.55，bootstrap 90% CI (0.025, 0.155)，P=**0.988**）。
- **Learning curve 检查**（round1-9 表上做 5折×5次重复分数据比例 group-CV）：

  | frac | n_pairs | MAE | R² |
  |---:|---:|---:|---:|
  | 0.25 | 7,020 | 1.842 | 0.765 |
  | 0.50 | 14,040 | 1.797 | 0.781 |
  | 0.75 | 21,061 | **1.770** | **0.796** |
  | 1.00 | 28,081 | 1.773 | 0.655 |

  0.25→0.75 单调下降（~4% 相对提升），尚未平台；1.00 时 R²/RMSE 明显恶化但判定为
  幸存者偏差（frac=1.0 时没有采样自由度排除重尾难例如含磷异常值）而非新的不稳定
  信号——round6 自己的学习曲线是同样形状。**总体判断**：round10 的边际收益是"温和
  但真实"，不是"必须做"的强信号，边际 MAE 收益外推约 0.01-0.03。round10 是否要跑
  留给用户决策，本节配方（`HANDOFF_round10_20260721_ZH.md`）写就待命。

- **2026-07-21 GPU 空闲时段的 side track：3D-GNN 架构探索**（`gnn3d_*` 系列脚本，
  DGT 启发）。在 round1-9 规模的一个 matched 子集（n=28,508，因几何可用性收窄）上
  对比四种架构：

  | 架构 | GNN-only MAE | best blend MAE |
  |---|---:|---:|
  | attentive2d（2D 基线，matched） | 2.460 | 2.188 |
  | attentive2d_v2（+距离层） | 2.329 | 2.179 |
  | attentive3d（真实 3D 坐标） | 2.386 | 2.186 |
  | distattn（距离注意力，最 3D-aware） | 2.295 | **2.178**（w=0.5，最佳） |

  **四个 blend MAE 挤在 2.177-2.188 一个很窄的带内**（n_test=188，小样本，差异大概率
  在噪声内）——**引入真实 3D 几何信息相对纯 2D+attentive 架构没有清晰的额外收益**。
  这是后续（09-04）"标签噪声天花板"结论的一个早期独立信号：当时就已经暗示，在这个
  规模和这套特征下，更复杂的几何表征已经榨不出更多信号了。

### 3.4 2026-07-20~29（约）：Snellius scratch 全量 purge —— 重大数据/权重损失

具体触发时间未精确记录（`benzoin_dg_repo_moved_and_data_loss` memory + `RECOVERY_REPORT_20260902.md`
详述），影响：

- **BDE 侧**：`aldehydes_all.csv` 局部 3D 描述符从 220k 行退化到仅 42,335 行部分重算；
  `products_all.csv`（产物局部描述符）**整个丢失，无备份**；B6 champion 的两个
  checkpoint（`.pt`，训练产出但被仓库级 gitignore 排除、从未有 home 备份）**永久丢失**。
- **cross 侧**：round8/9 的全部 DFT-SP 标签丢失；醛侧 r2SCAN-3c SP 缓存
  (`data/raw/dft_sp_funnelv3/`) 整个丢失、无备份；所有 cross GNN 的 `.pt` 权重丢失。
- **教训**（写进 memory `scratch-disk-quota-risk` 和多份 STATUS 文档）：
  **凡是同时被 `.gitignore` 排除、又没有 home 目录备份的产出物，一次存储策略变更
  就会永久消失**——这条教训此后驱动了"结果一律 `git add -f` + 加进
  `submit_backup_recovery_artifacts.sh` 归档清单"的纪律，以及后来的增量几何归档
  设计（BDE 侧改成 tar.zst 压缩保留而不是删除）。
- 仓库迁移：旧 `/gpfs/scratch1/shared/schen3/benzoin-dg` 变成空壳，live 仓库迁到
  `/gpfs/scratch1/shared/schen3/benzoin-dg-restored`，分支 `agent/recovery-20260902`。

### 3.5 2026-09-02~03：三线并行恢复

- **BDE**：`assemble_homo_descriptor_libs.py` 重算描述符库（正在进行，见 §4）；
  产物 BDE 标签从 home 备份 `bdfe_gxtb_products.tar.gz`（1460 块）**bit-exact 重拼**
  出 `products_bdfe_gxtb_descriptors.csv`（218,966 行）。
- **cross rounds1-7 复现验证**：`train_r17`（job 26314357）完成，**CV MAE 1.877 vs
  历史 1.883，全部落在噪声内** —— round1-7 确认为可信基线（memory
  `rounds17-reproduction-confirmed`）。
- **round10 特征组装 + AL 打分**完成（`assemble_r10feat`、`score_r10`），
  round10 候选池 15,983 行 × 583 列特征表就绪。
- **round8/9 DFT 标签从零重算**（09-03，用户："两个都做，按顺序"）：产物几何
  从 home 备份 `cross_round{8,9}_xyz_geometry.tar.gz` 解包直接用；**醛几何决定
  重新生成**（不从 BDE 窗口的 220k 库里 harvest——太慢/太脆/被窗口 A 进度卡住），
  一趟 `cb_featurize.py --homo-from --aldehydes-only --emit-aldehydes` 自包含产出
  几何+G_xtb+能量。~41.5k 次 r2SCAN-3c 单点能全重算（20,934 产物 + 20,617 唯一醛）。
  过程中修了 3 个 bug：g-xTB 二进制路径硬编码、resume-guard 只数行不看 error 导致
  27-chunk 缺口、`aldehydes.csv` 的 `index` 列语义混淆（chunk 内位置 vs 全局 lib_id）。
  **最终**：r8 7937 / r9 12823 对带标签（失败率 <0.1%）。
- **所有 cross GNN 从零重训**（`.pt` 权重全丢）：r1-7 recovered 版重训显示
  **GNN 在这个规模上无用**（gnn-only 2.368 vs ensemble-only 1.560，blend w_gnn=0.00）
  ——不是 bug，是数据规模问题（对比历史同规模 round7 也是类似的小样本高方差），
  真正的验证要等 r1-9/r1-10 规模。
- **inode 危机**（2026-09-03 深夜 22:35，硬顶 133%，账号级 EDQUOT）：根因是 BDE
  几何归档器追不上 220k 全库 featurize 阵列产生的散 xyz 文件（~600 chunk 积压
  ≈ 25 万 inode）。应对：BDE 两个 array 节流骤降（genoa 40→8，rome 100→15）、
  账号级孤儿清理器（清理被抢占任务残留的节点本地 scratch）、几何归档改为持续
  tar.zst 压缩（取代原计划的 `rm -rf`，因为几何要留给 Phase-3 3D 模型用）。
  凌晨 01:05 降到 123%（回到硬顶以下，写操作恢复），04:00 回落到 95%。

### 3.6 2026-09-03 晚~09-04 凌晨：隔夜自主流水线（用户："有序推进，一段时间不给指令"）

全链自动化，无需人工干预直到明确的决策点为止：

```
醛 SP 阵列(26354346) → assemble(26354407) → r1-9 组表+relabel+剪枝+训练(steps1-4)
  → r1-9 GNN(CPU) → bootstrap验证 → round10 用r1-9重打分 → round10产物SP选top-2000
  → STOP（round10 DFT / N值 是用户决策点，自动化不越界）
```

过程中修了 2 个新 bug（inode 危机之后）：`prune_table_to_champion_features.py` 的
META 列表漏了 `new_scaffold_split`（导致 `train_scaffold_disjoint.py` KeyError）；
`submit_r19_retrain_chain.sh` 的 `--ensemble-path` 文件名与实际产出不符。

**09-04 01:36 — r1-9 champion+ensemble 重训完成**：骨架不相交 holdout（n=448）
ensemble MAE **2.254**（R² 0.771），单XGB 2.544，g-xTB 基线 5.037。

**09-04 02:26 — r1-9 GNN+blend 完成**：GNN-only 2.313，**blend（w=0.40）MAE 2.167**
（bootstrap 90% CI [0.032, 0.140]，**P=0.9954**）——GNN 在 r1-9 规模上显著有效
（对比 r1-7 recovered 的 null，多 2 轮数据让它翻盘）。

**09-04 03:45 — 隔夜链完整跑完**：round10 用 r1-9 重打分产出新的 2000 对选择集
（取代旧的、基于 r1-7 的过时版本），round10 产物 SP 阵列跑上（1994/2000 完成）。
自主任务到此为止，主动停下等 09:00 用户接手。

同一窗口内额外发现了 3 个 purge 后脚本漂移 bug（全部 commit 修复）：GNN
`--ensemble-path` 命名约定不一致、09-01 旧 r1-9 目录残留会被误读、`prune` META
列表缺列。

### 3.7 2026-09-04：round10 落地 + r1-10 champion 确认

- **09:15** round10 醛+产物两腿 DFT-SP 全部完成（产物 1994/2000，醛 3166/3172），
  组装出 **1,969/1,994 对**带标签的新数据（`dG_orca_kcal` mean 6.36，p50 6.13，物理
  上合理）。
- **09:30** r1-10 champion+ensemble 重训：clean-train 22,771，骨架不相交 holdout
  ensemble MAE **2.326**（R² 0.739）——比 r1-9 的 2.254 略差，判定需 bootstrap 确认
  是否噪声。
- **10:07** r1-10 GNN+blend+bootstrap 完成：

  | 指标（holdout n=448） | r1-9 | r1-10 |
  |---|---:|---:|
  | ensemble-only MAE | 2.254 | 2.326 |
  | GNN-only MAE | 2.313 | 2.313 |
  | **blend MAE** | **2.167** | **2.215** |
  | bootstrap 90% CI | [0.032, 0.140] | [0.047, 0.181] |
  | P(blend>ensemble) | 0.9954 | **0.99855** |
  | w_gnn | 0.40 | **0.50** |

  **判读**：两代 blend 都显著优于 ensemble-only（P>0.995），GNN 权重还在上升
  （0.40→0.50）——blend 架构在更大数据规模上持续稳健。r1-10 vs r1-9 在这个 n=448
  冻结集上的差异（2.167 vs 2.215）落在噪声内（bootstrap SE ~0.15）；round10 那
  1,969 个 AL 选出的高不确定性对，按设计不会被这个冻结集采样到，所以它没有让
  这个特定 holdout 变锐——这**不代表 AL 没用**，需要一个 round10-aware 的评估
  （见下）。

- **10:15 — champion 确认为 r1-10**（`CHAMPION.md` 写就）。理由：数据更多（+1,969
  对 DFT 覆盖，含 AL 难例）、blend 优势更强、"最新+更多数据"原则；r1-9 冻结留作
  fallback。

- **10:15 — round10-aware AL 评估（第一层）**：r1-9 blend 在 round10 那 1,969 个
  AL 选出的对（对 r1-9 来说是真正未见过的 held-out）上 MAE = **2.780**（RMSE 3.86），
  比 r1-9 自己的冻结集 2.167 **差 28%**——**AL 确实挑出了模型的真实盲点**。
  不确定性信号本身有效：Pearson r(不确定性, 真实误差) = 0.40（p=9e-77），按
  不确定性四分位分组 MAE 单调从 Q1 2.245 到 Q4 3.798。

- **10:30 — round10-aware AL 评估（第二层，关键的 null 结果）**：
  `eval_round10_al_benefit.py` —— 把 round10 的 30%（589 行）冻结成硬测试集，
  分别用"不含 round10"（A）和"含另外 70%/1,370 行 round10"（B）训练同一个
  ensemble：

  | test | g-xTB | A（不含r10） | B（含r10） | B−A |
  |---|---:|---:|---:|---:|
  | r10_hard_test (n=589) | 5.216 | 2.793 | 2.760 | **−0.033** |
  | frozen_448 | 5.037 | 2.254 | 2.284 | +0.030 |

  两个 delta 都在噪声内（MAE bootstrap SE ~0.12-0.15）。**结论：round10 这一批
  AL 数据训练后对模型没有可测量的改善。** AL *诊断*出真实弱点（第一层），但
  *训练*那 1,370 个点几乎不改变精度。可能原因：1,370 行只是在 20,812 行基础上
  +6.6%，边际；AL 难例可能是"不可约噪声难"而非"覆盖难"；~21k 对规模下这套
  260 维特征已接近平台期。**推荐：不要再做 ~2k 规模的 AL 轮**，要么标签本身
  升级、要么换更大批次/更好特征/换问题形态。

### 3.8 2026-09-04：标签质量三探针（判断"还能不能推 MAE"）

r1-10 blend MAE 2.215 vs homo 项目已测过的单构象 r2SCAN-3c 标签噪声地板
（~2.14 kcal，K=5 conformers，34 分子）——如果 cross 的地板类似，说明**模型已经
到顶**，AL/更多数据都帮不上忙，该转向部署/换问题形态。三个探针同时用 fat_rome
空闲算力跑：

1. **`confnoise_cross`（构象噪声地板）**：32 个随机产物，各 K=5 个 ETKDG 构象
   → xTB opt → r2SCAN-3c SP → 每分子 DFT 能量的 std。**结果（32/32 完成）**：
   `dG_std_kcal` mean **2.975** / median **2.884**；single-conformer vs
   Boltzmann-average 标签的差异 mean **−0.147**（|mean| 仅 0.15 kcal）。
   **判读**：单构象标签噪声地板 ~2.9 kcal，champion MAE 2.215 已经贴着这个地板
   ——**多构象 Boltzmann 重标签是 null**（homo 项目上这个结论已经 2 次独立确认过，
   这是 cross 项目的独立第 3 次确认）。
2. **`geom_bias`（GFN2 vs g-xTB 几何的产物侧能量偏差）**：12 个杂原子重
   （硼/超价硫/磷）的产物，在同一起点分别做 GFN2-xTB opt 和 g-xTB opt，
   r2SCAN-3c SP 对比。**结果（12/12 完成）**：mean **−2.55**、median **−1.54**
   （|mean| 3.21 kcal），且**结构化**而非散点：超价 P(=O) 上 −9.2、多磺酰基 −5.0、
   硼 −4.2，非超价硫 ≈0。**判读**：GFN2 几何系统性地让超价磷/多磺酰基/硼的
   构型停留在偏高能量的构型上——这是一个真实的、有化学机理的、局限于特定官能团
   子集（~20-30%）的标签偏差，原则上可以靠"对这批分子换用 g-xTB 精修几何"来修正。
3. **`wb97x_shift`（泛函层级偏差，r2SCAN-3c vs wB97X-3c）**：❌ 第一个结果就是
   +3.98e6 kcal 的垃圾数值——`_parse_orca_energy` 解析不了 wB97X-3c 的 range-separated
   杂化输出格式（或者 ORCA 6.1.1 下 wB97X-3c 需要不同关键字）。判定不值得为此
   单独修解析器（泛函偏差本来就是三个候选杠杆里验证优先级最低的一个），取消。

**综合判读**：探针 1（构象噪声）说明"多算几个构象重标签"没用；探针 2（几何方法）
说明"换几何方法"对特定杂原子子集有真实、结构化的影响，但探针 2 只测了产物单侧
的能量，没有扣掉 donor/acceptor 侧的抵消——**真正决定性的测试是下面第 §3.9 节的
三物种 ΔΔG 消融**。

### 3.9 2026-09-04：决定性几何-方法消融（`dg_geom_method`，**09-04 当晚已收官**）

**目的**：探针 2 只看了产物一侧的能量偏差；但 `dG = G(prod) − G(donor) − G(acc)`
是个差值，如果三个物种的几何偏差方向一致，donor/acceptor 侧的偏差可能会**抵消**
掉产物侧的偏差，让最终标签 ddG 几乎不变。这是"值不值得花几千次 DFT 重新做定向
几何重标签"这个决策的**唯一决定性证据**。

**方法**：60 对（35 个杂原子重 = hetero 组 + 25 个对照 = ctrl 组），每对的 donor/
acceptor/product 三个物种都：固定同一个 ETKDG 起始构象 → 分别做 GFN2-opt 和
g-xTB-opt（**g-xTB 从 GFN2 极小点出发精修，不独立重新搜索**——这是个关键设计，
下面会解释为什么）→ r2SCAN-3c SP → `dG_gfn2` 和 `dG_gxtb` 两套三物种 ΔG →
`ddG = dG_gxtb − dG_gfn2` = 抵消donor/acceptor之后剩下的、纯几何方法造成的标签
偏差。同时记录每个物种的 RMSD(GFN2构型, g-xTB构型) 做健全性检查。

**v1 版的一个真实 bug（已修）**：第一版（`26369906`）让 GFN2 和 g-xTB 各自从一个
共享的 ETKDG 种子**独立**优化，结果第一行 control 分子的 `rmsd_prod` 就有 0.87 Å
——两个优化器落进了**不同的构象**，测出来的 ddG 是构象噪声，不是几何方法偏差，
整个探针被confound了。修复：g-xTB opt 现在强制从 GFN2 的极小点出发做**精修**
（refine，不是 re-search）——这和探针 2（`geom_bias`）一直用的方式一致。修复后
重跑（先是 `26369972`，后因 ORCA MPI 单点在这台机器上跑不通又改回串行、最终版
`26371408`）。

**最终结果**（59/60 对完成，1 对已知 stale-schema 分片缺失，不影响判读）：

| 组 | n | 中位 ddG (kcal/mol) | 分布范围 |
|---|---:|---:|---|
| hetero（杂原子重） | 34 | **−0.47** | −5.22 ~ +2.84 |
| control | 25 | **−0.25** | −4.43 ~ +1.96 |

`rmsd_prod_med = 0.20 Å`（<0.3 健全性阈值，确认"从 GFN2 极小点精修"修复生效，
不是构象噪声在冒充几何偏差）。两组中位数都远低于 1 kcal 判定阈值、同号、分布
高度重叠——**判读走§预案的第二支：没有显著几何方法偏差**。

**判读预案（原写在 `HANDOFF_20260904.md` §1.3，供后续会话执行；结果已按此判读）**：
- 若 hetero 组 ddG 中位数 >1 kcal、明显 >> control 组、且 rmsd 干净（<0.3）→
  对 r1-10 表里杂原子重的 ~5,000-8,000 对（`smiles` 含 `P(=O)`/`S(=O)(=O)`/硼酯）
  做定向的 g-xTB 几何精修 + r2SCAN 重标签，用新标签覆盖，重训，在一个
  B/S/P 富集的 holdout 上对比新旧 champion。
- **若没有偏差 → 几何方法不是杠杆，标签质量investigation 到此为止，全力转向
  Goal 3**（部署 / 换问题形态，见下节）。**← 实际结果，09-04 晚间执行**：未启动
  任何定向重标签战役，champion 维持 r1-10 不变，资源全部转向下节的 Goal 3 交付。

### 3.10 2026-09-04：Goal 3 —— 重表述为分类/排序（已验证为正，可立即交付）

**动机**：即使 §3.9 的结果是正面的，靠几千次定向 DFT 重标签能把 MAE 往下推多少
也有限（label floor 本身在 ~2.9），继续死磕"预测精确的 kcal/mol 数字"这个指标
本身收益递减。一个自然的问题：如果只需要"哪些反应对更favorable"这种排序/分类
信息（很多实际筛选场景真正需要的就是这个），champion 现有的连续预测是否已经
足够好，只是被 MAE 这把尺子低估了？

**方法**：**不重新训练任何模型**，直接拿冻结的 r1-10 champion blend 在它自己
n=448 骨架不相交 holdout 上的连续预测值，做两种事后重表述：(a) 在一个阈值上
二值化成 favorable/unfavorable，算分类指标；(b) 直接看预测排序质量。同时和
g-xTB 物理基线做同样的重表述，看 ML 相对基线的优势在排序/分类视角下是被放大
还是缩小。脚本：`cross_benzoin/eval_reformulation_classification_ranking.py`。

**结果**（回归 MAE 2.215 精确复现，脚本正确性自检通过）：

| 指标 | ML (champion blend) | g-xTB 物理基线 | 随机基线 |
|---|---:|---:|---:|
| Spearman rho（预测 vs 真值排序） | **0.884** | — | 0 |
| T=0 kcal 二分类（16% 正例） AUC / acc / F1 | **0.934 / 0.911 / 0.697** | 0.781 / 0.752 / 0.448 | 0.5 |
| T=训练集中位数 4.92 二分类（46% 正例） AUC / acc / F1 | **0.924 / 0.848 / 0.835** | 0.726 / 0.632 / 0.689 | 0.5 |
| 真实最优 10% 中，模型 top-10% 命中率 | **0.644** | 0.311 | ~0.10 |
| 真实最优 20% 中，模型 top-20% 命中率 | **0.756** | 0.511 | ~0.20 |

**判读**：尽管连续值 MAE 卡在标签噪声地板上动不了，champion 的**排序/分类质量
非常好**（AUC 0.92-0.93，两种阈值下都稳健），且在每一个切面上都**远超 g-xTB
物理基线**（不是"和基线差不多、只是换了个指标显得好看"，是真实的、大幅的
判别力提升）。这是一个**立刻可以交付、不依赖 §3.9 结果、不受标签噪声天花板
束缚**的 Goal-3 产出——一个"favorable/unfavorable 筛选器"或"按预测 ΔG 排序
取 top-k 候选"的工具，今天就能可靠地用，无需再等更多 DFT 数据或更精细的模型。

**09-07 更新：从"验证过的发现"变成"用户能拿到手的工具"**。09-04 的产出还只是
一个独立的事后评估脚本（`eval_reformulation_classification_ranking.py`）；09-07
把这三个结论直接做成了 `predict_dg.py` 输出 CSV 里的标准列：`dg_favorable`
（dG<0，物理意义最直接的判据）、`dg_below_train_median`（更高召回率的判据）、
`dg_rank_pct`（批内百分位排名，用于给一批候选分子对排序）。任何拿 `predict_dg.py`
打分新分子对的人，现在默认就能拿到这三列，不需要另外知道/运行评估脚本。部署
过程中顺带发现并修复了一个严重 bug，见 §4.5。

### 3.11 champion 演化全景表（一图看懂 10 轮）

| 版本 | 时间 | clean-train 行数 | ensemble-only MAE | GNN-only MAE | **blend MAE** | 备注 |
|---|---|---:|---:|---:|---:|---|
| R1-3 | 07-14~15 | ~2,472 | — | — | 2.966(CV)† | 旧切分，类别多样性采样 |
| R1-4 | — | — | — | — | 2.261(CV)† | |
| R1-5 | — | — | — | — | 2.633† | 首次三编码器GNN，P=0.987(n=29) |
| R1-6 | 07-16 | — | — | — | 2.582† | GNN未复现，P=0.456(n=29) |
| R1-7（旧切分） | 07-17前 | ~27,583前身 | — | — | 1.883(CV)† | screen10k耗尽 |
| **R1-7（骨架不相交重估）** | 07-17 | 19,687 | 2.256 | — | **2.215** | 泄漏勘误后的诚实数字 |
| R1-8 | 07-20 | 27,583 | 2.201 | 2.324 | **2.106** | attentive-pooling GNN 引入，P=0.9923 |
| R1-9（purge前） | 07-21 | 43,367 | 2.163 | 2.162 | **2.074** | 学习曲线支持继续，P=0.988 |
| **R1-9（purge后重算）** | 09-04 | ~32,630 | 2.254 | 2.313 | **2.167** | 标签从零重算，P=0.9954 |
| **R1-10（当前champion）** | 09-04 | 22,771 | 2.326 | 2.313 | **2.215** | +round10 AL批次，P=0.99855 |

（† 见 §3.2 勘误；purge 前后的 r1-9/r1-10 由于底层 DFT 标签重算、骨架切分重建，
数字不直接可比，只在各自内部前后一致。）

**十轮下来最重要的方法论收获**：
1. **诚实的评估比好看的数字更重要**——骨架泄漏一次性让所有历史数字回撤 0.2-0.5 MAE；
   BDE 子课题独立踩到同一个坑（见 §4），说明这不是偶然。
2. **GNN blend 的收益随规模显现**——round1-7 是 null，round1-8 开始稳健显著，
   一路到 round1-10 权重还在涨（0.40→0.50）。"小数据上架构对比得出的 null 结论"
   本身不能外推到更大规模。
3. **AL 能诊断盲点，不保证能修复盲点**——round10 是本项目第一次把这两件事分开
   测量，结果是"诊断阳性、修复阴性"，这本身是个重要的方法论产出，提醒未来不要
   把"不确定性排序有效"直接等同于"训练后会变好"。
4. **标签噪声地板是硬约束**——当 MAE 逼近标签本身的噪声 std 时，继续加同类数据
   / 加模型容量都会边际收益归零（round10 AL null + 07-21 三种3D架构挤在一起，
   两条独立证据链在 09-04 汇合成同一个结论）。
5. **重表述目标可以绕开标签噪声天花板**——分类/排序视角下同一个模型远没有触顶。

---

## 4. 子课题：BDE 预测（键解离能）

### 4.0 定位

预测反应中间体的键解离能本身（不是当作 cross 项目的输入特征），两个目标键：
醛 formyl **C–H** BDE（缩合第一步）、产物中心 **ketC–carbC** BDE（缩合关键 C–C 键）。
标签均为项目自己的 g-xTB 计算，r2SCAN-3c/CPCM(DMSO) 几何一致。

### 4.1 Phase 1（2026-07 之前~07-20）：七个基线跑完，B6 夺冠

| 模型 | 醛 MAE/R² | 产物 MAE/R² | 一句话 |
|---|---|---|---|
| **B6 GNN+局部3D描述符融合（冠军）** | 1.104/0.900（旧split）| 2.076/0.921（旧split）| D-MPNN 图嵌入 + H-SPOC 局部电子结构描述符经 `x_d` 通道融合 |
| B4 D-MPNN（纯2D图） | 1.604/0.830 | 3.641/0.834 | |
| B5 BonDNet式（反应差分图） | 1.923/0.802 | 3.689/0.846 | |
| H-SPOC（局部3D+XGB） | 2.42/0.758 | 4.05/0.818 | 零新增计算，性价比最高的基线 |
| D-SPOC-217（全局描述符差） | —/0.629 | 5.365/0.715 | |
| B2 ECFP+XGB | 3.49/0.563 | 6.18/0.652 | 传统指纹基线 |
| B0/B1 ALFABET | 近乎无区分力 | 输出方差趋零 | 域外分子，退化成预测均值 |

### 4.2 与 cross 项目平行的骨架泄漏发现（独立确认同一个方法论教训）

用 molecule-cold-split 测出的旧数字（醛 1.104、产物 2.076）事后被查出
**87.6%/83.9% 的测试集 scaffold 也出现在训练集里**，数字虚高。重训到真正骨架
不相交切分：

| 划分 | 醛 MAE/R² | 产物 MAE/R² |
|---|---:|---:|
| molecule-cold-split（已废弃） | 1.104/0.900 | 2.076/0.921 |
| **真·scaffold-disjoint 重训（对外一律引用）** | **1.579/0.843** | **3.060/0.886** |
| 相对退化 | +43% | +47% |

**跨子课题独立确认**：cross 项目 07-17 发现 +0.2~0.5 MAE 的泄漏溢价，BDE 项目
独立发现 +43%~47% 的泄漏溢价——两个不同任务、不同代码、相近方法学，**都在同一
类骨架泄漏问题上栽了跟头**，说明这不是某一次实现的巧合，而是"分子级别切分不等
于化学结构泛化"这个更普遍教训的两次独立验证。（一个有意思的反差：cross 项目自己
重训后 R² 反而略微**改善**——泄漏对不同模型/任务的影响方向不能假设可迁移，必须
逐个实测。）

**其他关键方法论结论**：
- `x_d` 融合（图嵌入+局部3D描述符拼接）有效——复用自主 dG 模型已验证过的模式。
- **homo→cross 迁移**：数据规模够大时"仅 cross 训练"会追平"homo 预训练+cross微调"
  ——round1-7 规模时 C（预训练+微调）4.501 vs A（仅cross）4.524，round5 规模时
  差距收窄到噪声量级，round1-7 规模基本消失。实用结论：round8 起直接 cross-only
  训练即可。
- g-xTB 标签的系统性误差主要来自 **single-point 方法层级**，不是几何优化（DFT
  仲裁 round2：Δ_SP 均值 −21.0 vs Δ_geom 均值 1.58）——要真正缩小和 DFT 的
  ~14-16 kcal 差距，得升级 single-point 方法本身。
- g-xTB 标签的构象噪声下限 **~2.1-2.7 kcal/mol**（单构象）——和 cross 项目
  09-04 测出的 ~2.9 是同一类物理限制的两次独立测量，量级一致。
- nitro 官能团的双峰 Δ_SP 后来查明是 **3/18 个 g-xTB 自身电子结构失败**（醛
  formyl C-H BDE 算出 >100 kcal/mol，物理上不合理），不是真正的 level-of-theory
  双峰——用 `bde_gxtb_kcal` 本身作判别量即可识别。
- **homo dG 难尾巴 = g-xTB 基线失败，不是模型/标签问题**：150 个最差 P/磺酰基/
  亚胺/酰胺分子，|g-xTB基线误差|均值 13.9 kcal（全库 4.3），corr(残差,基线误差)
  = 0.888 ——推理时应按子结构强制 route_to_dft。
- **homo dG 主动学习不会提升**：pool-AL 不适用（库已全标注）；难尾巴 Boltzmann
  重标签是 **2× 独立确认的 null**（冻结 MAE 10.01→10.13，boltz_corr 与残差正交）
  ——和 §3.8 cross 项目的构象重标签 null 是**跨子课题的第 3 次独立确认**同一个
  现象：Boltzmann 重加权不能修复 DFT 标签噪声。

### 4.3 2026-07 purge 后资产盘点

| 文件 | 状态 |
|---|---|
| 醛 BDE 标签（220,522行） | ✅ bit-exact 恢复（home 备份逐块重拼） |
| 醛局部 3D 描述符 `aldehydes_all.csv` | ⚠️ 仅部分重算 42,336 行（原 ~220k） |
| 产物 BDE 标签 `products_bdfe_gxtb_descriptors.csv` | ✅ 09-02 从 home 备份重拼（218,966 行） |
| 产物局部 3D 描述符 `products_all.csv` | ❌ 无备份，需全量重算 |
| B6 checkpoints（`.pt`） | ❌ **永久丢失**，只能重训 |

**结论**：当前无法一键复现 B6 champion——醛侧标签有但描述符只有 42k，产物侧
标签和描述符都缺。

### 4.4 2026-09-02 至今：全量重建 + inode 危机 + 本会话接管

- **09-02** 启动 220,859 个 homo pair 的全量 featurize array（`26316404` genoa
  2209 task，一趟同时产出产物+醛两个局部描述符库）。float-id 恢复 bug 修复
  （`"2.0"` vs `"2"` 静默 join 到 ~0 行，加 `qc.norm_id()`）。
- **09-03 深夜** 该阵列是 inode 133% 危机的根因（几何归档跟不上产出速度），
  两个阵列被临时节流（40→8，100→15）。
- **09-04 ~14:50 — 本会话接管整个项目**（用户："全项目由本会话统一驱动"，
  取代此前"窗口 B 别碰 BDE"的分工）：
  - 发现 `bde_geomarch`/`clean_orphans` 两个常驻服务撞上 fat_rome 24h 墙钟正在
    过期（不会自动重投），已重投（`26373559`/`26373560`）。
  - inode 已回落到 **19.8%**（远低于 90% 的"可以调回"阈值），把两个 array 的
    节流恢复到危机前水平（genoa 8→**40**，rome 15→**100**），验证 20 秒内
    running 数如期跳变。
  - 新起监控 `bbzu8ozjc`：轮询两个 array 完成状态；同时看护 geomarch/janitor
    两个服务的存活，连续 2 次探测不到就自动重投（间隔 ≥90 分钟防抖）。
  - （09-04 撰写时的进度快照：featurize 80%，已被下面 09-06/09-07 的收官结果
    取代，不再单列。）
  - `bde_post_sweep`（`26326313`）挂 `afterany` 依赖，阵列落地后**全自动**触发：
    完整度门槛检查 → 重组两个描述符库 → 备份 → 提交完整 model sweep
    （B6 checkpoint 重训 + B6 5-seed deep ensemble + B4/B5 诚实重训 + GBM 头
    bake-off 全量重跑）。计划文档 `pipeline/bde/POST_ARRAY_MODEL_SWEEP.md`。

- **09-06 — 两个 featurize array 全部跑完，`bde_post_sweep` 全自动落地**：
  genoa 2191 完成/18 失败，rome 1191 完成/68 失败（失败率均 <6%，可接受）。
  自动装配出**史上第一次完整**的两个描述符库：`aldehydes_all.csv`（209,526
  行，从 42k 局部库补全）、`products_all.csv`（184,199 行，此前从未有过全量
  版本）。自动提交并跑完 4/5 项计划的 model sweep（B6 ckpt、B6 deep ensemble、
  B4/B5 honest retrain、GBM bake-off；第 5 项 Phase-3 3D 反应差分模型的自动
  提交漏掉了，见下）。

- **09-07 — 本会话补完 sweep 的收尾工作，产出新 BDE champion**：sweep 脚本
  自动跑完了训练，但没有自动跑聚合/文档/备份这几步，本会话补上：
  - 跑 `aggregate_b6_ensemble.py`（原来 `ensemble_summary.json` 是空的）→
    **新 champion：B6 5-seed deep ensemble，醛 MAE 1.851 / 产物 MAE 2.826**
    （单 seed checkpoint 2.094/3.192，deep ensemble 领先 ~12%；不确定性
    `sigma~|err|` spearman 0.40-0.42，可用于 route-to-DFT 分流）。
  - **这组数字取代 09-04 之前引用的 1.579/3.060**——那是在 42k 局部库上跑的，
    数据规模不同，不是同一个实验，不能直接比较优劣，但作为"现在能引用的
    champion 数字"，官方以新数字为准。
  - 更新 `pipeline/bde/STATUS.md`、`git add -f` 所有结果 json/pred csv/
    checkpoint、扩充 `submit_backup_recovery_artifacts.sh` 备份清单并跑了
    备份 job。
  - **Phase-3 3D 反应差分模型（计划第 5 项）：查证后决定不跑**。追查脚本
    (`gnn3d_schnet_dimenet.py`) 发现它其实预测的是 `dG_orca-dG_gxtb`（cross-dG
    的目标），不是 BDE 目标，仓库里没有任何 BDE 版本的 3D 脚本；而这个架构族
    （SchNet/DimeNet++/ViSNet）07-21 已经在 dG 任务上跑过、4 个架构全部落在
    MAE 2.177-2.188（§3.11 提到的 null 结果之一）。要做成真正的 BDE 版本是
    一次实打实的重写（换目标、拆几何包、修复两处路径/环境损坏），先验价值
    不高，没有无人值守启动，写进了 `STATUS.md` 供以后参考。

### 4.5 2026-09-07：一次跨子项目的 bug——BDE 重建意外打断了 cross-dG 的部署工具

**背景**：`data/cross_benzoin/homo_v6/aldehydes_all.csv` 是 BDE 和 cross-dG 两个
子项目**共用**的文件——BDE 的 `assemble_homo_descriptor_libs.py` 写它，cross-dG
的 `predict_dg.py`（部署工具，见 §3.10）读它。09-06 的 BDE 全量重建只计算了
BDE 自己需要的局部键描述符，**从没重新算过整分子的 g-xTB 自由能 `G_gxtb`**
（BDE 自己的目标是键解离能，从不需要这个量）。

**发现经过**：09-07 落实 Goal 3 部署前，没有直接宣布"部署完成"，先拿 20 个真实
分子对冒烟测试了 `predict_dg.py`——结果**对任何新分子对都返回 0 行、直接崩溃**。
追查发现：`load_round()` 用 `donor_G_gxtb`/`acceptor_G_gxtb` 是否非空作为"这一行
有没有匹配上醛库"的判据；由于 09-06 后这个字段对 207k/209k 个醛都是空的（只有
2,718 行从旧的 42k 局部库回填过），这个判据把**每一行真实成功的匹配都误判成
失败**，整批直接清零——SMILES 层面的匹配其实完全正常（手工验证过）。

**修复**（commit `c520acf`）：把判据换成覆盖率 99.9%+ 的 `donor_xtb_HOMO`/
`acceptor_xtb_HOMO`（和 G_gxtb 缺口无关），真实匹配的行不再被误删；`G_gxtb`（260
个冠军特征里的 2 个）该缺照缺，走已有的中位数填补兜底，和其余可选特征的处理方式
一致。20/20 重新冒烟测试全部通过。**这个 bug 不影响 r1-10 champion 本身的 MAE
2.215**——那是在重建前的快照上训的——只影响"用 `predict_dg.py` 给全新分子对
打分"这条路径，而这正是 Goal 3 要交付的东西，不测出来的话就是一次"看起来部署了、
实际打不出分"的假交付。

**后续（同日）：把 G_gxtb 缺口本身也补上**。没有满足于"median-impute 兜底就够
了"，而是写了一个廉价的补算方法：复用已经优化好、已经归档的几何（不重新做
昂贵的构象搜索+优化+Hessian），只在这个固定几何上补两个便宜的单点能（GFN2 SP
+ g-xTB SP），套用项目已有的混合修正公式重建 G_gxtb。小样本验证（97 个分子）：
0.54 秒/分子，和旧库里唯一重叠的分子对比只差 0.10 kcal/mol。2200-chunk 补算
array（`26432805`，节点选择见下）已提交，跑完后 `merge_aldehyde_gxtb.py` 会把
结果并回库里。

**顺带的运维发现**：提交补算 array 前查了集群节点状态（`sinfo -s`）——**genoa
分区当时整个 drain/down**（737 个节点全部不可用，不是单纯排队），`fat_genoa`
大半 down，`gpu_h100` 全部 drain。据此把新作业排到了负载较低的 `rome`，避免
提交到一个完全瘫痪的分区。

---

## 5. 子课题：homo dG（A+A 同源，已上线，简述）

- **状态**：已上线、已验证，不在本次总结的主动学习叙事主线内（本项目对 homo
  的主动学习已经做过、结论是 pool-AL 不适用+ Boltzmann 重标签 2×null，见 §4.2）。
- **覆盖**：219,364 个 DFT 标签，覆盖全部 220,859 个同源醛库。
- **champion**：`pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`，
  测试 MAE **1.503**。推理入口 `benzoin_dG.predict_dG_champion()`。
- **与 cross 线的关系**：两条独立的模型线，不是一个统一通用模型；"同源+交叉合并
  训练"测试过两次（4,120 行规模 −1.9%，17,270 行规模 −1.0%），方向一致但收益随
  cross 自身数据增长而萎缩，是本项目一直保留但优先级不高的开放问题。
- 难尾巴诊断（g-xTB 基线失败）、AL null 结果见 §4.2，是与 cross 项目 §3.8 相互
  印证的证据链的一部分。

---

## 6. 贯穿全程的基础设施事故

| 事故 | 时间 | 影响 | 应对 |
|---|---|---|---|
| **git 对象库损坏** | 2026-07-13 | 历史重开（"Fresh history"），此前的提交记录丢失（代码现状保留） | 无法挽回，此后本项目全部历史从这个提交开始计 |
| **Snellius scratch 全量 purge** | 2026-07-20~29（约） | B6 checkpoint 永久丢失、round8/9 DFT 标签丢失、大量中间产物丢失、所有 cross/BDE GNN 权重丢失 | 2026-09-02 起系统性恢复：能从 home 备份重拼的重拼，能重算的重算（标签、几何、GNN 全部从零重训），建立"结果必须 `git add -f` + home 归档"的新纪律 |
| **scratch1 inode 硬顶 133%** | 2026-09-03 深夜 22:35 | 账号级 EDQUOT，写操作全面受阻，影响 BDE+cross+NHC 三个并行工作 | 根因诊断（BDE 几何归档跟不上）→ 节流两个 array → 账号级孤儿清理器 → 几何改增量压缩归档；01:05 回落到硬顶以下，04:00 到 95% |

**共同教训**（写进多份 memory/STATUS 文档）：
1. gitignore + 无 home 备份 = 一次存储策略变更就永久丢失，任何"训练产出但不
   打算长期保留在 scratch"的东西都要么 `git add -f`、要么进归档清单。
2. 共享有限资源（scratch inode、GPU fairshare、SLURM QOS）上的自动化流水线，
   必须内建监控 + 节流开关，不能假设资源无限。
3. 交接文档需要"随会话失效必须重建"的显式提醒（本项目 09-03 起固定了
   HANDOFF 的标准结构，见 `HANDOFF_20260904.md` §8）——纯 SLURM/git 状态之外，
   会话内的 Monitor 任务不会跨会话存活。

---

## 7. 当前状态快照（09-04 ~15:40 首版，**09-07 ~13:00 更新**）

### 7.1 09-04 原始快照（历史存档）

| 线 | 状态 |
|---|---|
| **cross-benzoin ΔG champion** | r1-10 blend，MAE 2.215，`CHAMPION.md` 冻结 |
| **cross 主线在跑** | `dg_geom_method 26371408`（50/60，决定性几何偏差消融，rmsd_prod_med 0.20 健康）|
| **cross Goal 3 已交付** | 分类/排序重表述验证为正（AUC 0.92-0.93）；`predict_dg.py` 端到端预测 wrapper 已做 |
| **BDE featurize 重建** | 1759/2209（80%），节流已恢复到危机前水平，ETA 数小时 |
| **BDE 自动化** | `bde_post_sweep(26326313)` 挂依赖，阵列完成后全自动跑完整 model sweep |
| **homo dG** | 稳定在线，本轮无新动作 |
| **git** | 分支 `agent/recovery-20260902`，HEAD 领先 origin ~92 commit（本 repo 全部历史），全部未 push，等三线收齐统一 push |
| **inode** | 19.8%，安全 |

### 7.2 09-07 ~13:00 现状（本次增补撰写时刻）

| 线 | 状态 |
|---|---|
| **cross-benzoin ΔG champion** | 仍是 r1-10 blend，MAE 2.215，未变 |
| **几何偏差消融** | ✅ 已收官，无显著偏差，见 §3.9 |
| **cross Goal 3** | ✅ 已交付为工具：`predict_dg.py` 默认输出分类/排序三列（见 §3.10），且修复了一个曾让它 100% 失败的 bug（见 §4.5） |
| **BDE champion** | ✅ 全量重训完成，新数字醛 1.851 / 产物 2.826（deep ensemble），见 §4.4 |
| **BDE G_gxtb 补算** | 🏃 array `26432805`，871 完成/10 失败/18 运行中（约 40%，rome 分区繁忙拖慢速度，ETA 未知，非阻塞） |
| **homo dG** | 稳定在线，本轮无新动作 |
| **git** | 已 push 到 origin（09-07 首次），HEAD 领先历史全部保留，之后的新 commit 也已同步推送 |
| **inode** | 17.3%，安全（09-04~09-05 之间有一次 94.4% 峰值的自愈事件，见 `RUN_LOG_20260903.md`） |
| **集群健康** | genoa 分区 09-07 完全 drain/down，新作业改投 rome；fat_rome/rome 仍是可用的健康分区 |

**周末计划回顾（原 09-04 制定）**：
1. ~~等 `dg_geom_method` 出结果 → 若确认偏差可修复，人工搭建定向重标签战役~~
   → **实际结果：无偏差，未启动战役**，见 §3.9。
2. ✅ 已交付分类/排序重表述，09-07 进一步做成了工具默认输出。
3. **仍未**盲目扩大 AL 批次——判断维持不变。
4. ✅ BDE model sweep 已在 09-06 自动跑完，09-07 补完了聚合/文档/收尾，产出新
   champion（见 §4.4）。**新增的、原计划外的工作**：G_gxtb 跨项目 bug 修复
   （§4.5）、G_gxtb 全量补算（进行中）。

---

## 8. 文档地图（权威 living 文档，本总结之后请看这些）

| 文档 | 作用 |
|---|---|
| `RUN_LOG_20260903.md` | 逐步日志（09-03 起，所有 jobid/决策/bug，living doc） |
| `HANDOFF_20260904.md` | 最新交接（第 0-1 节 = 立即行动） |
| `HANDOFF_20260903.md` / `HANDOFF_20260902.md` | 更早的交接快照 |
| `cross_benzoin/CHAMPION.md` | 当前 champion 加载方式、数字、lineage |
| `cross_benzoin/docs/STATUS_ZH.md` / `STATUS_EN.md` | cross 项目滚动状态（截至 07-20，含骨架泄漏勘误全文） |
| `cross_benzoin/docs/HANDOFF_round10_20260721_ZH.md` | round10 完整配方 + 决策证据 |
| `pipeline/bde/STATUS.md` | BDE 子课题权威入口（排名、purge 资产盘点） |
| `pipeline/bde/POST_ARRAY_MODEL_SWEEP.md` | BDE 阵列后 model sweep 计划 |
| `RECOVERY_REPORT_20260902.md` | 2026-07 purge 恢复全过程 |
| `data/cross_benzoin/reformulation_classification_ranking_eval.json` | §3.10 分类/排序重表述的完整数字 |
| `cross_benzoin/predict_dg.py` | Goal 3 部署工具本体，09-07 起默认输出分类/排序三列 |
| `pipeline/bde/recompute_aldehyde_gxtb.py` / `merge_aldehyde_gxtb.py` | §4.5 G_gxtb 补算方法 + 合并脚本 |
| memory `~/.claude/.../memory/*.md` | 见 `MEMORY.md` 索引，尤其 `cross-round10-fat20-stage1-recovered`、`cross-r1-10-champion-and-label-ceiling`、`bde-scaffold-leakage-finding`、`homo-active-relabel-null-result`、`bde-scaffold-disjoint-retrain-submitted`（09-06 数字更新）、`predict-dg-g-gxtb-regression-fixed`（09-07 新增，§4.5 的完整记录） |

---

*本文档由 Claude（Sonnet 5）在 2026-09-04 会话中撰写，覆盖至撰写时刻的全部
可验证历史（git log 92 commits、RUN_LOG/HANDOFF 全文、champion metadata、
memory 索引）。2026-09-07 应用户要求增补更新（见文首说明），补入 §3.9 最终
判决、§3.10/§4.4 的收官数字、新增 §4.5。后续进展请更新 `RUN_LOG_*.md`——
本文件原则上仍是时间点快照，只在有明确的汇报用途需求时才做增量编辑。*
