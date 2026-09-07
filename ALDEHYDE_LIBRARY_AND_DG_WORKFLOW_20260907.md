# 醛数据库构建历史 + cross-benzoin dG 工作流 / 描述符工程（2026-09-07）

> 英文版见 `ALDEHYDE_LIBRARY_AND_DG_WORKFLOW_20260907_EN.md`，两份改动时保持同步。
>
> 独立于 `PROJECT_SUMMARY_20260904.md`（那份是 AL 主动学习主线的时间线叙事）。
> 这份文档回答两个更"底层"的问题：**醛结构库是怎么来的、备份现状如何**，以及
> **dG 预测这条工作流具体长什么样、每一步的描述符工程细节**。技术参考性质，
> 不是时间线叙事，写完之后过时了就更新，不必按"快照"惯例整篇重写。

---

## 一、醛结构库（`aldehydes_clean_v6`）

### 1.1 来源与过滤流水线

**原始池**：约 45 万个分子（`name/SMILES/MW/CID/...`），来源是一个更早期就已经
存在的 PubChem 风格候选池（脚本注释里写的原路径是已经不存在的
`/scratch-shared/schen3/aldehydes.csv`，早于本仓库当前 `agent/recovery-20260902`
的 git 历史起点，无法再追溯到那次会话的原始下载/筛选记录——这是本库**唯一
没有留痕迹的一环**，之后每一步都有脚本和产出文件可查）。

**过滤脚本**：`pipeline/filter_smiles_v6.py`（v1→v6 迭代，v6 是当前版本，v1-v5
的脚本和产出都还留在仓库里，见 `data/library/aldehydes_clean_v{1..6}.csv`，
方便追溯每一版规则改动的影响）。v6 相对 v5 的改动（脚本 docstring 原文）：
1. 新增拒绝规则 `malformed_boron`——价态不完整的裸硼原子（如 `[B]-C`），这类
   PubChem 脏数据 GFN2-xTB 会给出物理不合理的能量（pilot 里见过 −237 kcal/mol
   的离谱值）。
2. `xtb_risk` 标签扩展到 boron/phosphorus/selenium——这三类元素在 v5 的 ΔG
   离群值里富集了约 11-12 倍，QM 不够可靠但仍然保留（不是拒绝，是打标签降权/
   加审），处理方式和已有的 nitro/azide/N-oxide 一致。

**拒绝分类（first-failing-rule-wins，按脚本原文顺序）**：
`invalid_parse`（RDKit 解析失败）→ `multi_component`（盐/混合物/抗衡离子）→
`net_charged`（净电荷非零）→ `mw_too_high`（>500 Da）→ `disallowed_element`
（元素不在 {H,C,N,O,F,S,Cl,Br,I,B,Si,P,Se} 内）→ `isotope`（同位素标记）→
`malformed_boron`（v6 新增）→ `zwitterion_or_ylide`（未被 nitro/azide/N-oxide/
胺氧化物等允许母题解释的两性离子）→ `not_single_aldehyde` / `multi_aldehyde`
（0 个或 >1 个真醛基）→ `enal` / `ynal`（α,β-不饱和醛/炔醛，走 NHC Stetter/
homoenolate 分流，不算标准苯偶姻底物）→ `alpha_dicarbonyl` → `reactive_group`
（亚硝基/偶氮/SF5/超价硫氟/烯酮/异腈）→ `vinyl_conj` → `aliphatic_too_large`
（无芳环且碳数 >12）。

**最终产出**：`data/library/aldehydes_clean_v6.csv` = **220,859 个醛**（library
里各处引用的 "220k"/"220,859" 就是这个数），`aldehydes_rejected_v6.csv` = 被拒绝
的约 72.8 万条（含拒绝原因），供审计/复核用。

### 1.2 备份 / 版本控制现状

**结论：这个文件本身是安全的，不需要额外操心。**

```
git log --oneline -- data/library/aldehydes_clean_v6.csv
25f5400 Fresh history: initial commit after old .git object database was found corrupted (2026-07-13)
```

它**从一开始就在 git 里**（`25f5400`，本仓库当前历史的第一个提交，早于 2026-07
git 数据库损坏事故之后的"重开历史"），从未被 gitignore，体积也不大（~41MB），
git 本身（+ 已推送的 GitHub 远程）就是它的备份，不依赖 home 目录的额外归档。
`v1`-`v5` 的历史版本和 `aldehydes_rejected_v{1..6}.csv` 同样都在 git 里，随
`data/library/` 整个目录一起追踪。

**这和下一节的局部/全局描述符库（`aldehydes_all.csv` 等）形成鲜明对比**——那些
文件因为体积更大、增长更快，被 gitignore 排除，purge 时全部丢失，是本项目
2026-09 恢复工作的主要内容。

---

## 二、局部/全局描述符库的构建与恢复历史

结构库（§一）只是 SMILES + 拒绝标签；真正喂给模型的是在这份结构库基础上，对
每个醛跑几何优化 + QM/描述符计算之后产出的第二层文件，全部放在
`data/cross_benzoin/homo_v6/`。这一层文件全部 gitignore（`*_all.csv` 之类），
**这正是 2026-07 scratch 全量 purge 造成灾难性损失的那一层**。

### 2.1 文件清单与当前状态（2026-09-07）

| 文件 | 内容 | 行数 | 当前状态 |
|---|---|---:|---|
| `aldehydes_bdfe_gxtb_descriptors.csv` | 醛 BDE（键解离能）标签，g-xTB 单点 | 220,522 | ✅ bit-exact 从 home 备份重拼（`homo_v6_scratch_archive/bde/bdfe_gxtb` 逐块） |
| `aldehydes_mordred_slim102.csv` | 醛侧 mordred 描述符（102 列精选子集） | ~220k | ✅ bit-exact 从 home 备份重拼 |
| `aldehydes_all.csv` | 醛局部 3D 电子结构描述符（xtb/morfeus/multiwfn） | **209,526** | ✅ 09-06 全量重建（此前 purge 恢复只补了 42,336 行的局部子集）；`G_gxtb`（整分子 g-xTB 自由能）09-06 重建没算，**09-07 已补算完成**，覆盖率 99.07%（见 §六） |
| `products_bdfe_gxtb_descriptors.csv` | 产物 BDE 标签 | 218,966 | ✅ 09-02 从 home 备份重拼 |
| `products_all.csv` | 产物局部 3D 电子结构描述符 | **184,199** | ✅ 09-06 **史上第一次**全量版本（此前从未存在过全量版，purge 前也只有 partial） |
| `aldehydes_scaffold_split_from_dG.csv` / `products_scaffold_split.csv` | 骨架不相交 train/val/test 划分 | 220,859 全覆盖 | ✅ 一直在 git 里，未受 purge 影响 |
| B6 BDE 模型 checkpoint（`.pt`） | — | — | ❌ purge 中永久丢失原始权重，09-06 全量重训产出了新的（见 `PROJECT_SUMMARY §4.4`），已 `git add -f` |

### 2.2 2026-07 purge 的损失范围 + 恢复路径

- **丢失原因**：这一层文件体积大（百 MB 到 GB 级）、迭代快，按项目早期惯例
  `gitignore` 掉了，只存在于 Snellius 的 `scratch` 文件系统（当时路径
  `/scratch-shared/schen3/benzoin-dg`）。2026-07-20~29 前后 scratch 被全量清空，
  没有 gitignore-豁免、也没有系统性 home 备份的文件全部消失。
- **恢复靠的是运气 + 纪律的组合**：`aldehydes_bdfe_gxtb_descriptors.csv` /
  `aldehydes_mordred_slim102.csv` 之所以能 bit-exact 恢复，是因为**早前某次会话
  出于个人习惯把它们 tar 进了 `/home/schen3/benzoin_backups/`**（不是项目制度
  安排的，纯属偶然）；`products_all.csv` 完全没有这种偶然备份，2026-09-02 之后
  只能全量重算（成本约是醛侧的 ~6 倍，因为产物分子更大更复杂）。
- **2026-09-02 起系统性补救**：建立了 `cross_benzoin/slurm/
  submit_backup_recovery_artifacts.sh`——把这一类"gitignore + 训练产出，丢了就
  真丢了"的文件系统性 tar 进 `/gpfs/home4/schen3/benzoin_backups/recovery_20260902/`
  （home 目录，不受 scratch 生命周期策略影响），幂等、可随时重跑，09-07 又扩充
  了一次清单（加入 `products_all.csv` 本身和这次 model sweep 的结果目录）。

### 2.3 备份位置汇总（2026-09-07 现状）

| 备份目录 | 内容 |
|---|---|
| `/home/schen3/benzoin_backups/bde_homo_rebuild_20260902/` | `aldehydes_all.partial42k_20260902.csv.tar.gz`（09-02 局部重建的旧版，仅供历史对比）+ `aldehydes_all_full_20260906.csv.tar.gz`（09-06 全量重建版） |
| `/gpfs/home4/schen3/benzoin_backups/recovery_20260902/` | `submit_backup_recovery_artifacts.sh` 管理的系统性归档：醛库三件套、DFT-SP 标签、round8/9/10 中间表、champion 模型目录、09-06 BDE model sweep 结果（`runs/logs/scaffold_disjoint_bde/`） |
| git（`data/library/`、`*_scaffold_split*.csv`、champion 模型 `.joblib`/`.pt`、所有结果 `.json`/`_pred.csv`） | 体积可控的都直接 `git add -f` 进仓库，2026-09-07 已推送到 GitHub 远程，是最可靠的一层备份 |

**曾经唯一还没关闭的缺口**：`aldehydes_all.csv` 里的 `G_gxtb` 列（详见 §六），
不是"丢失"，是这次 09-06 重建从没算过。**2026-09-07 已补算完成**（覆盖率
1.3% → 99.07%），这条线收尾。

---

## 三、dG 预测工作流总览（端到端）

给定一对醛（donor + acceptor，可同分子）预测苯偶姻缩合反应的 `dG_orca_kcal`
（r2SCAN-3c DFT 水平），完整流程：

```
①醛结构库(§一) → ②geometry+局部描述符(cb_featurize.py, funnel_v3搜索) →
③mordred全局描述符(免费的醛侧 + 现算但便宜的产物侧) → ④DFT单点标签(仅训练集需要) →
⑤拼表(assemble_cross_training_table_v3.py) → ⑥裁剪到冠军260特征
(prune_table_to_champion_features.py) → ⑦模型推理(predict_cross_champion.py:
MLP+XGB tabular ensemble ⊕ triple-encoder attentive GNN, blend w_gnn=0.5) →
⑧部署封装(predict_dg.py, 见 PROJECT_SUMMARY §3.10/§4.5)
```

### 3.1 几何搜索方法：funnel v3

`pipeline/compute/conf_funnel_v3.py`（在 v2 基础上加一层拓扑守卫）：

- **v1→v2 的改进动机**：早期 K3→K10→K20→K30 收敛性研究发现 DFT ΔG 标签有
  ~1.4 kcal 的构象搜索噪声 + ~4% 的灾难性失败（6-15 kcal 量级），根因是 v1 用
  `numThreads=0` 做 RDKit 构象嵌入——多线程嵌入即使固定随机种子也不确定，导致
  独立跑几次会漏掉真正的全局极小点。v2 修复：`numThreads=1 + randomSeed=42`
  （完全可复现）+ RMSD 剪枝（保留的构象是不同的势阱，不是近似重复）+ 采样密度
  加倍。
- **v2→v3 的改进动机**：CREST A/B 测试发现 funnel 唯一的**灾难性**失败模式是
  最低能构象**连接性被破坏**（GFN-FF 把构象弛豫成了异构化/断裂的结构，用它算
  出的虚假低能量主导了 Boltzmann 平均，实测一例 ΔG 因此偏了 12 kcal）。CREST
  自带拓扑检查能避免这个，但它的 GFN-FF metadynamics 对复杂苯偶姻这类大分子
  采样不够密。所以不换引擎，只给 funnel 补上它缺的这一层：**每个构象做一次
  便宜的 RDKit 键连接性感知，和输入图的连接性不一致的直接丢弃**，其余流程
  （密集 ETKDG 采样 → GFN-FF 预筛 → 保留 top-L → GFN2 优化 → Boltzmann 排序）
  不变。
- **标准流程**（v1 起未变）：dense ETKDG 采样 → GFN-FF 快速预筛 → 保留 top-L
  （默认 L=10）→ GFN2 `--opt` 精修 → 按 GFN2 能量排序，取最低者为代表构象。

### 3.2 标签来源

- **真实标签 `dG_orca_kcal`**：ORCA r2SCAN-3c 在 funnel_v3 几何上做单点。只有
  被 AL 选中做过 DFT 的分子对才有（训练/评估用）。
- **物理基线 `dG_gxtb_kcal`**：g-xTB single-point，同样在 funnel_v3 几何上，
  比 DFT 快几个数量级，全量 220k pair 都有；既是模型的一个输入特征，也是报告
  里"ML 相对物理基线提升多少"的对照组。
- **醛侧整分子自由能 `G_gxtb`**（区别于上面的 pair-level `dG_gxtb_kcal`）：
  `gxtb_baseline.py` 建立的混合修正配方——GFN2 `--ohess`（优化+Hessian）拿到
  热力学修正 `(G_gfn2 − E_gfn2)`，g-xTB 只做单点电子能 `E_gxtb`（g-xTB 在本
  项目里没有生产级热力学修正流水线），`G_gxtb = E_gxtb + (G_gfn2 − E_gfn2)`。
  这是 09-06 BDE 重建漏算、09-07 正在补算的那个量，见 §六。

---

## 四、描述符工程细节

### 4.1 局部电子结构描述符（xtb / morfeus / multiwfn）

醛侧（`ALDEHYDE_FEATS`，27 个原始量，见
`cross_benzoin/assemble_cross_training_table.py`）和产物侧（`PRODUCT_FEATS`，
38 个）用**同一套方法**算，字段基本对称，方便模型学到跨物种一致的表示：

| 层 | 具体量 | 说明 |
|---|---|---|
| xtb 单点/派生 | `xtb_energy`、`xtb_HOMO`/`xtb_LUMO`/`xtb_gap`、`xtb_IP`/`xtb_EA`（vertical，`--vip`/`--vea`）、`xtb_mu`/`xtb_eta`/`xtb_omega`（化学势/硬度/亲电指数，从 IP/EA 派生）、`xtb_dipole` | 标准反应活性描述符 |
| Mulliken 电荷 | 关键原子（醛基 C/O，产物侧 ketC/ketO/carbC/hydO/hydH） | 定点电荷，不是全分子 |
| WBO（Wiberg 键级） | 醛基 C=O；产物侧新生成的 C-C 键 + 原有羰基键 | 键强度/反应中心指示 |
| Fukui 指数 | `fukui_plus`/`fukui_minus`/`fukui_0`/`dual`（有限差分，`_fukui_finite_diff`） | 亲核/亲电位点识别，取关键原子处的值 |
| 质子亲和能 `pa_*` | 醛基氧上人工加一个 H，算能量差 | 碱性/氢键受体强度的代理量 |
| morfeus：buried volume `vbur_*` | 关键碳原子周围的立体拥挤度 | |
| morfeus：Sterimol `sterimol_L/B1/B5` | 立体位阻的长度/宽度描述符 | |
| morfeus：`SASA_total`、`P_int` | 可及表面积、极性积分 | |
| Multiwfn：ADCH 电荷、QTAIM（键临界点的 Laplacian/椭率） | 仅 `aldehydes_all.csv`（本轮才补上，见 §六表格里的 `adch_*`/`qtaim_*` 列） | B6 BDE 模型的 `x_d` 融合特征之一，dG 项目目前不用 |
| 仅产物侧：氢键几何 `hb_dist`/`hb_angle`/`dih_core` | 反应新形成的分子内氢键（β-羟基酮的经典构象）+ 核心二面角 | 产物特有，醛侧没有对应量 |
| 醛侧独有：整分子自由能 `G_xtb`（GFN2）/ `G_gxtb`（g-xTB） | | §三.2 已说明；`G_gxtb` 是当前的补算对象 |

### 4.2 全局分子描述符（mordred + RDKit 2D）

- **mordred**（`pipeline/analysis/finalize_correction_mordred_slim.py` 定的
  targeted families：MoRSE / CPSA / Polarizability / GeometricalIndex /
  MomentOfInertia / PBF / McGowanVolume / VdwVolumeABC / Weight / TopoPSA——
  不是 mordred 全量 1800+ 描述符，是精选过的子集）：
  - **醛侧完全免费**：`aldehydes_mordred_slim102.csv`（102 列）来自已有的全量
    220k 库 mordred 计算，donor/acceptor 各自查表拼接，零新增计算。
  - **产物侧是真实的新增计算，但很便宜**（`add_mordred_cross_products.py`，
    ~1 秒/分子）：复用已经保存的产物几何（`xyz_file`），不需要重新优化，纯
    后处理。
- **RDKit 2D**（`RDKIT_FEATS`，16 个：`MW/LogP/TPSA/HBD/HBA/RotBonds/ArRings/
  ArHetRings/AlRings/Rings/Heteroatoms/FractionCSP3/BertzCT/Kappa2/
  NumStereocenters/n_CHO`）：donor/acceptor/product 三边各算一份，零成本
  （纯拓扑，不需要几何）。

### 4.3 pair-level 交互特征（尝试过，冠军模型没保留）

`assemble_cross_training_table_v3.py` 会生成 `interaction_gap_HOMOd_LUMOa`
（donor HOMO − acceptor LUMO，前线轨道匹配）、`interaction_fukui_match`
（donor 亲核 Fukui × acceptor 亲电 Fukui）、以及 `MISMATCH_PAIRS`
（`xtb_gap`/`dipole`/`sterimol_L/B1/B5`/`SASA_total`/`MW`/`TPSA` 各自
donor-acceptor 差的绝对值，衡量"这对底物的电子/立体互补性"）——**这批特征是
带着"cross 独有的、homo 预训练学不到的信息"这个假设专门设计的，但实测冠军
模型的冻结 260 特征列表里一个 `interaction_*` 都没有**（验证见下节），说明
这批显式交互特征没能在特征选择/裁剪阶段挤进最终模型——图神经网络那一半
（GNN blend 分量）大概率已经隐式学到了同等或更强的交互信息，让这些手工构造
的显式交互项变得多余。这是一个值得记录的负结果，避免以后重新"发明"同一批
特征。

### 4.4 冠军模型实际用的 260 个特征构成（`feature_list.json` 实测统计）

| 分组 | 数量 | 说明 |
|---|---:|---|
| `donor_*` | 71 | 43 个局部 QM 原始量（§4.1）+ 28 个 mordred |
| `acceptor_*` | 82 | 43 个局部 QM 原始量 + 39 个 mordred |
| `product_*` | 69 | 16 个 RDKit 2D + 53 个 mordred |
| 产物侧局部 QM（无前缀，`xtb_*`/`fukui_*`/`mulliken_*`/`wbo_*`/`sterimol_*`/`dual_*`/`vbur_*`/`hb_*`/`pa_*`/`SASA_total`/`P_int`/`dih_core`） | 35 | §4.1 表格里"仅产物侧"那一批 |
| `bde_gxtb_kcal` | 1 | pair-level g-xTB BDE 基线（来自 `cross_round*/bde_gxtb/`，和 §三.2 的醛侧 `G_gxtb` 是两个不同的量，不要混淆） |
| `interaction_*` | **0** | 见 §4.3，尝试过但没留在最终列表里 |
| **合计** | **260** | |

mordred 总计 120/260（46%），是最大的单一来源，RDKit 2D 50/260（19%），
局部 QM 原始量（含 `donor_G_gxtb`/`acceptor_G_gxtb` 这两个，见 §六）约 90/260。

---

## 五、模型架构（`predict_cross_champion.py`）

**Blend = tabular ensemble ⊕ triple-encoder GNN**，两半独立训练，推理时线性
混合（`blend_w_gnn` 从 metadata.json 读取，当前冠军 r1-10 是 0.50）：

- **Tabular 半边**：`MLPXGBEnsemble`（`train_cross_ensemble.py`）——一个 MLP
  + 两个 XGBoost（"XGB-a"/"XGB-b"，大概率是不同随机种子/超参的 bagging 变体）
  的简单平均，吃上面裁剪好的 260 个特征。
- **GNN 半边**：`TripleGNN` / `TripleGNNAttn`（`train_cross_gnn.py` +
  `gnn_architectures.py`）——**三个独立的图编码器**，donor 分子图 + acceptor
  分子图 + product 分子图各一个，当前冠军用的是 attentive-pooling 变体
  （`TripleGNNAttn`，07-20 引入，比默认池化在骨架不相交划分下有稳健优势，见
  `PROJECT_SUMMARY §3.11`）。
- **推理环境**：单一 `envs/gnn_lite`（torch + torch_geometric + sklearn +
  xgboost + rdkit 都在一起），不需要跨环境 subprocess 桥接（homo 项目的
  ENSEMBLE72 模型因为环境冲突需要桥接，cross 项目这里刻意避免了同样的问题）。
- **不确定性代理**：`predict_dg.py` 用的是 tabular 半边内部 3 个基学习器
  （MLP/XGB-a/XGB-b）预测值的标准差，是一个**廉价、只看方向对不对**的代理量，
  不是完整的 pair-grouped bootstrap 认知不确定性估计（那个更贵，见
  `score_round_active_learning.py --n-boot`）。

---

## 六、已收尾的缺口：`G_gxtb`（醛侧整分子 g-xTB 自由能）

**背景**（2026-09-07 撰写时）：09-06 的 BDE 描述符库全量重建只算了 BDE 自己
需要的局部键描述符，从没重新算过 §三.2 定义的整分子量 `G_gxtb`——这个量是
`donor_G_gxtb`/`acceptor_G_gxtb`（260 个冠军特征里的 2 个）的来源，重建后
209,526 个醛里只有 2,718 个（从旧的 42k 局部库回填）有值，其余 ~207k 是空的。

这个缺口**曾经导致 `predict_dg.py` 对任何新分子对 100% 失败**（一个覆盖率
哨兵字段选错导致误判"没匹配上库"，实际匹配是成功的）——已修复，详见
`PROJECT_SUMMARY_20260904.md §4.5`。修复之后这两个特征只是被正常
median-impute，不再让整行数据蒸发。

**补算已完成**（不是重新做整个几何搜索，只是在已有的优化几何上补两个便宜的
单点能，见 `pipeline/bde/recompute_aldehyde_gxtb.py` 和 `PROJECT_SUMMARY
§4.5` 的方法说明）：**2026-09-07**，array `26432805` 跑完（2175/2200 chunk，
25 个已知不可恢复），`pipeline/bde/merge_aldehyde_gxtb.py` 把结果并回
`aldehydes_all.csv`（只填缺失的 `G_gxtb`，不动已有的 2,718 行）——覆盖率
1.3% → **99.07%**（207,580/209,526）。残留 ~0.93% 来自那 25 个不可恢复
chunk，接受不再追。合并后 `predict_dg.py` 20-pair smoke test 干净。

---

## 七、关键脚本 / 文件索引

| 脚本/文件 | 作用 |
|---|---|
| `pipeline/filter_smiles_v6.py` | 醛结构库过滤（§一） |
| `data/library/aldehydes_clean_v6.csv` / `aldehydes_rejected_v6.csv` | 结构库产出 |
| `cross_benzoin/cb_featurize.py` | 单对分子的端到端 featurize 入口（几何+局部描述符） |
| `pipeline/compute/conf_funnel_v3.py`（+ `v2`） | 几何搜索方法（§3.1） |
| `pipeline/compute/gxtb_baseline.py` | `G_gxtb` 的原始混合修正配方（§3.2） |
| `cross_benzoin/add_mordred_cross_products.py` | 产物侧 mordred（§4.2） |
| `cross_benzoin/assemble_cross_training_table_v3.py` + `assemble_cross_training_table_combined.py` | 拼表（§4.3-4.4），09-07 修过一个 bug（§六引用） |
| `cross_benzoin/prune_table_to_champion_features.py` | 裁剪到冻结特征列表 |
| `cross_benzoin/train_cross_ensemble.py` / `train_cross_gnn.py` / `gnn_architectures.py` | 模型训练（§五） |
| `cross_benzoin/predict_cross_champion.py` | 推理封装（模型层） |
| `cross_benzoin/predict_dg.py` | 端到端部署工具（用户层，见 `PROJECT_SUMMARY §3.10`） |
| `pipeline/bde/recompute_aldehyde_gxtb.py` / `merge_aldehyde_gxtb.py` | `G_gxtb` 补算（§六） |
| `cross_benzoin/slurm/submit_backup_recovery_artifacts.sh` | 描述符库家族的 home 备份清单（§2.3） |
| `data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` | 冻结的 260 特征列表（本文档 §4.4 统计的来源） |

---

*首版 2026-09-07。技术参考文档，不是时间线快照——过时了直接改，不需要另开
新版本。项目整体进展/时间线请看 `PROJECT_SUMMARY_20260904.md` 和
`RUN_LOG_20260903.md`。*
