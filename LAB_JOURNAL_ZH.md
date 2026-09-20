# 实验日志 —— 苯偶姻 ΔG 预测

> 每步操作的有理有据叙述：**尝试了什么、为什么是对的一步（原理）、哪里出错 + 根因、
> 结果。** 图放在有帮助处。持续更新；每天的小节晚上收尾。
>
> 2026-09-10 按用户要求开始。English: `LAB_JOURNAL.md`（保持同步）。电报体日志是
> `RUN_LOG_20260903.md`；组件地图是 `PROJECT_PLAN_ZH.md`。2026-09-10 之前的历史在
> `RUN_LOG_20260903.md` 和 `HANDOFF_*.md` / `PROJECT_SUMMARY_*.md`，此处不详细回填。

---

## 2026-09-10

### 当天开始时的背景
两个战役并行：
- **cross Tier B**（全部 35,528 个 cross 对的自洽 B97-3c 重标注）—— "便宜基线杠杆"：
  若把 Δ-learning 基线 g-xTB → B97-3c 降低残差 scatter（pilot 说 std 4.32 → 1.11），
  冠军 MAE 可能首次跌破 2.9 kcal 标签噪声地板。当天 ~68–72%。
- **homo 从零重标注** —— 用户指示：在*全*库上、用重算的 DFT 标签（2026-07 purge 丢的
  ~18.9 万个）重建 homo（A+A）模型，用单个 XGBoost + 单个 GNN。

### 尝试 1 —— homo 重标注走"SP-on-归档几何"快线
**原理。** homo 全 DFT 标签物理丢失（已核：不在 git、不在新旧 scratch、不在 87 个 home
备份 tarball 里）。但**几何**（GFN2-opt，2026-09 BDE-featurize 归档）和 **xTB 热校正**
（存在 `products_all.csv` / `aldehydes_all.csv`）活了下来。所以不需要贵的全流程（构象
搜索 + xТБ Hessian + DFT）—— 只在归档几何上做**一个 DFT 单点**，复用存储的热校正：
`G_lvl(物种) = E_lvl(归档几何上的 SP) + (G_xtb − E_el_xtb)_存储`。这正是 2026-06 原始
homo 标签的做法（整个 21.9 万库 ~18 h）。比自洽 regen 便宜 ~10–20×，且和描述符库几何
一致。

建了：`build_homo_sp_manifest.py`（184,052 对）、`homo_sp_from_geom_worker.py`（从
`geom.tar.zst` 抽出产物 + 醛 xyz，每物种跑 r2SCAN-3c + B97-3c，`dG = (G_prod −
2·G_ald)·627.509`）、`submit_homo_sp.sh`（CHUNK 20 → 9,203 阵列任务）。起 2 臂
（fat_genoa + genoa）。

### 问题 1 —— smoke：3 对全 `sp_fail`，ORCA 返回 None
**根因。** `thermo_orca.calc_orca_sp` 把它的 scratch 写成输入 xyz *旁边*的 `orca_sp/`。
worker 在**同一个**抽取出的几何文件上并发跑多个 SP（产物 r2SCAN-3c 和产物 B97-3c 都指
`geoms/xyz_p000062.xyz`），共用一个 `orca_sp/` 目录、互相覆盖 `input.inp` / 输出 → 全挂。
**修。** `_sp` 现在每次调用把几何复制到自己的 `tempfile.mkdtemp()` —— 已验证的
`dft_sp_from_geom.py` 就是这个模式。之后对一个归档几何直接 `calc_orca_sp` 跑干净
（E = −1575.285 Eh，6.7 min）。已提交。smoke #2 过：3 个 QC 对，repro vs 存储 30k 标签
+0.06 / +8e-6 / +2.9 kcal —— 无系统偏移。

### 问题 2 —— `merge_homo_sp.py` 对 498 个部分 shard：判决 RED，27% 的 dG 是垃圾
对至今做完的 ~4,200 对跑 merge/QC。
- `repro_r2scan` **中位 1.3 kcal**（很好 —— 大多数对复现了幸存的 DFT 标签），但
  **std 6 × 10⁵**，且 **27% 的 |dG| > 200 kcal/mol**（到 ±10⁶）。
- 中位没问题 = 大多数抽取正确；一个大尾巴灾难性错误。

**根因（拉最差的行诊断）。** `_extract` 把每个抽取的成员缓存到
`dst / member.replace("/","_")` —— 即按**成员 basename** `xyz/p000062.xyz`。但 `pNNNNNN`
是**每 chunk 的本地序号**，不是全局 id：chunk_1523 的 `p000062` 和 chunk_1900 的
`p000062` 是不同分子。一个任务处理 20 个*打乱的*来自随机 chunk 的对；第一个要本地序号
62 的对填了缓存，之后每个要*不同 chunk* 的 `p000062` 的对拿到第一个分子的几何。结果：
对错分子算 E → 能量差 ~10× → dG 爆掉。E_prod/E_ald 原子数比即使对垃圾行也 ≈ 2.0，因为
两个错分子大致是对的*大小*类 —— 所以光靠大小检查抓不到。

**修。**
1. 缓存键现在含 chunk 目录：`<chunk_NNNN>__<member>`。
2. 抽取后**校验 xyz 原子数 == SMILES 重原子+H 数**（`_n_heavy_h`），不符即拒。防御纵深
   —— 也兜住归档缺口和任何未来的成员混淆。
取消两臂，**wipe 全部 ~430 个污染 shard**，重启。~7 h 计算作废 —— 但 merge QC 在任何
建模之前抓住了它，这正是对部分数据跑 QC 的意义。

### 问题 3 —— smoke #3（修好抽取）：仍有一行 dG = 28,293 kcal/mol
chunk-keyed + 原子检查的 worker 的 20 对 smoke。19 行干净（dG 1–7 kcal，对的分子），
**1 行垃圾**。E_prod/E_ald 比是 2.0 且原子数匹配 —— 这次几何*对*。

**根因。** `dG = (E_prod + thermal_prod) − 2·(E_ald + thermal_ald)`。E_prod ≈ 2·E_ald 时，
垃圾必来自**热校正**。manifest 的 `thermal_ald = G_xtb − xtb_energy` 取自
`aldehydes_all.csv`。查那列：`thermal_ald` 对 ~8% 的行范围 **−84 Ha 到 +87 Ha**（物理
热校正是 +0.05 到 +0.9 Ha），且有些行两个化学上不同的醛共享同一个 `G_xtb` 值。所以
`aldehydes_all.csv` 的 `G_xtb` 列对 **~8% 的行 id 错位** —— 2026-09-06 BDE featurize
烙进去的缺陷（home 备份里也是这些坏值，所以不是 purge 损坏）。`xtb_energy` 本身看着正常。

丢失的热校正没有干净的来源（`products_all.csv` 的 `G_donor` 也是从那坏列填的）。

**修。** worker 现在**界检存储热校正**（起初 `0 ≤ thermal ≤ 1.6 Ha`）；不过就**在归档
几何上重算热校正**，用一个新鲜的 `xtb --ohess`（`_thermal_job`，折进同一个作业池）。
新 `th_src` 列记录每物种 `stored` vs `recomp`。便宜（小醛 ~5–15 min）、自洽（和存储值同
GFN2 层级）。

### 问题 4 —— smoke #3（热校正就地重算）：27% → 小比例仍垃圾
界检 `0 ≤ thermal ≤ 1.6 Ha` 太松。一个具体的漏网：醛 `O=Cc1c(O)ccc(C(F)(F)F)c1Br`
（18 原子）*存储*热校正 = **1.229 Ha**（应 ~0.13）。1.229 < 1.6 所以过了，它以
`−2·thermal_ald` 进 ΔG → ~1500 kcal/mol 误差。
**根因。** Gibbs 热校正随原子数近线性（这些有机物 ~0.003–0.012 Ha/atom：13 原子醛
0.072，26 原子 0.136）。一个*绝对*界分不开合法的大分子热校正和错位的小分子热校正；
一个**每原子**界可以。
**修（commit `b4706df`）。** `_th_ok(v, n_atoms)` 要求 `0.001 ≤ v/N ≤ 0.020`（N = SMILES
重原子+H）。一个本身出界的重算热校正被拒。加一个硬门：worker 处任何 `|dG_r2scan| > 200
kcal/mol` → `error=dG_implausible`，垃圾永不进 shard。已验证该界拒 1.229 Ha 值
（0.068/atom）、过合法的 0.13（0.0072/atom）。smoke v4（`26554455`）在验证。
*注：* 残留 ~1–3% 的对（如构象面巨大的双硼酸苯偶姻）仍会有噪声大的单构象标签 —— 那些
是用户点出的"库设计上含算不出来/难算的例子"；merge 的 `repro_r2scan` 尾巴 + 组表时的
|dG| 截断处理它们，我们不追。

### 两轨拆分
`split_homo_sp_manifest.py` 扫每个归档、把 184,052 对拆：
- **archived 轨 —— 179,431（97.5%）**：几何在 → `homo_sp_from_geom_worker`（归档几何上
  SP；坏热校正经 `_thermal_job` 自愈）。
- **regen 轨 —— 4,621（2.5%）**：几何缺（chunk ~1400–1830 只部分归档）→
  `submit_homo_regen.sh` 跑完整 `rec_homo_relabel_worker`（conf_funnel_v3 + `--ohess`
  新鲜热校正 + 每物种 r2SCAN-3c + B97-3c + g-xTB）。每对慢但只 4.6k 个。
merge 时对比 regen 子集 vs archived 子集的标签分布看有没有系统偏移（regen 那条路早先
2 对 smoke 显示 vs 存储标签 ~−5.5 kcal —— 可能是真的构象改善或协议差异；若真，带一个
`is_regen` 标志）。

### 集群分享（用户指示）
用户："别占满所有节点 —— 和另一个项目（NHC）分享。"NHC 动力学战役在同账号 / QOS 上。
动作：Tier B fat_genoa 两臂降到 %8（正在跑的任务 scancel —— resume-safe，每个丢 ~1 对，
腾出 ~250 个 fat_genoa 节点），rome 臂到 %48；homo 重标注以温和并发重启，只随 Tier B
腾空间才上调。存成 memory `share-cluster-nodes`。NHC 的 `nhc-gsp-fine-*` 确认恢复正常。

**傍晚续 —— 用户第二轮输入：**
1. **催化剂空间不用我管** —— 我不需要关心催化剂，只关心底物。→ PROJECT_PLAN §3 精简为
   "不在范围"。
2. 上面说的 md 都要中英文两份。→ 建 `PROJECT_PLAN_ZH.md` / `CHEMICAL_SPACE_ZH.md` /
   `LAB_JOURNAL_ZH.md` / `CLUSTER_SHARING_ZH.md`。
3. 和隔壁对话商量好队列/节点规则 + 科学的提交方式；如果不是必须用 fat、只是因为
   genoa/rome 拥挤而用 fat，要想清楚合理性和必要性。→ 起草 `CLUSTER_SHARING.md`
   （提议：fat_genoa 归 NHC；benzoin-dg 只用 rome/genoa；fat 靠内存需要而不是拥挤来用；
   每分区稳态 ~90 上限留 ~40 余量）。**把 Tier B 从 fat_genoa 移回 rome**（取消
   26467948/26468648，range 1800-4440 以 %40 重投到 rome `26554708`；rome 臂 26467946
   → %50）。fat_genoa 现在完全归 NHC。（NHC 会话此时已重启，联系不上；能联系上就发提议。）

### 项目重新奠基（用户输入，当天已落实）
1. *homo* 库 = `aldehydes_clean_v6.csv`（220,860），过滤后 —— 且它**刻意包含无法优化/
   DFT 的分子**。那部分损耗是库属性，不是 bug；"算不了"是有效的记录在案的结果。
2. *cross* 库**建错了**。`candidates_v3`（~124 万对）是一个任意构造的子集，不是空间。
   **真实 cross 空间是 220,860² ≈ 4.88 × 10¹⁰ 有向醛对**。
3. 不落地 SMILES。建一个 **"flying dataset"** —— 惰性、按对地址的虚拟数据集：冻结
   220,860 个醛的 canonical `ald_idx`，保留每醛缓存（几何 / QM / Mordred / BDE / xTB
   能量），*按需*从两个醛缓存生成一个对的产物 SMILES + 260 特征 + 基线 + split。规范
   已写：**`CHEMICAL_SPACE.md` / `_ZH`**（冻结索引 → 重 key 缓存 → `pair(i,j)` 读取
   API → 标签/切分 → 迁移现有 35,528 标签 → 退役 `candidates_v3`）。构建顺序刻意小步，
   下游信任之前要做 bit 级特征复现校验。
4. **过去的 AL（1–10 轮）暂停；cross AL 会重做**（在 flying dataset 上）。35,528 个 DFT
   标签保留为数据；r1-10 blend（MAE 2.215）现在是*参照*模型，不是冻结的前进主线。
5. **催化剂空间**从姊妹 GitHub 仓库填了背景（`nhc-benzoin-pipeline` = TS_CC/TS_CN +
   微观动力学 ee(t)；`nhc-benzoin-active-learning` = ~1300 万立体异构体的多目标池式 AL，
   目标 = Kozuch–Shaik 能量跨度 + |ee|，都可删失；`stereo-catalyst-engine` = 贝叶斯优化
   催化剂设计；`nhc-pkah-predictor` = azolium pKaH 门槛）—— 但**本项目不管**。

### 当天交付物
- `PROJECT_PLAN.md` + `_ZH`（v6 库警示、§2.11 flying dataset、§2.7 AL 暂停、§3 精简为
  不在范围）。
- `CHEMICAL_SPACE.md` + `_ZH`（flying dataset 规范）。
- `CLUSTER_SHARING.md` + `_ZH`（集群分享规则提议）。
- `LAB_JOURNAL.md` + `_ZH`（本文件）。
- memory `working-style-journal-plan-understand`（journal + plan + 理解每一步）、
  `share-cluster-nodes`。

### 值得记住的原理笔记
- **对部分数据跑 QC 的意义。** 对战役 ~2.5% 跑 `merge_homo_sp.py` 是在 ~7 h（而不是
  ~3 天 + 一个训好的模型）之后抓住 27% 污染的原因。任何长标注跑都做便宜的部分 QC。
- **"E_prod ≈ 2·E_ald" 是好但不够的检查。** homo 对的产物 ≈ 两个醛单元，电子能接近醛
  的两倍。它抓离谱错配，但抓不到"错分子但对的大小"，也抓不到坏热校正。要原子数检查
  *和*热校正界检查*和* repro-vs-存储标签 QC，三个一起。
- **恢复 CSV 的数据质量是反复出现的隐患。** post-purge / post-featurize 的
  `homo_v6/*_all.csv` 已经咬了 4 次：float id、跨 chunk 几何序号复用、`G_xtb` 粗错位、
  `G_xtb` 细错位。**每个存储的 xТБ 量现在用前都界检**（每原子热校正带、|dG| 合理性门）。
  这比信文件、到模型训练时才发现便宜。

### 2026-09-11 —— 会话恢复、真实吞吐 ETA、flying dataset 第 1-2 步

**接续被打断的上午会话。** 真正工作仓库是 `benzoin-dg-restored`，不是环境默认给的
陈旧 `benzoin-dg` 壳。找到打断前未提交的工作 —— `assemble_homo_standalone_table.py
--full-library`（HANDOFF_20260910 §1.3 标的"待写"模式）+ `merge_homo_sp.py` 的双目录
glob —— 用磁盘上已有的部分 shard 冒烟测试通过（11,362 行×200 特征；merge 判决 GREEN，
无偏移），提交（`a0cdc17`）。重建了三个会话监控（guardian、Tier B drain、homo SP
drain）—— 它们不随会话存活，这是第三次照交接文档的重建配方重写了。

**"要不要上 fat 节点加速？"** 查 `sinfo -s`：`fat_rome`/`fat_genoa` 集群里都是 0
空闲节点——现在上去也不会更快，而且还比 rome/genoa 贵 50%（正是昨晚撤出的理由）。真正
的杠杆是 rome 的空闲（121 个空闲节点，我们和 NHC 都没在那抢）——把 Tier B 两条 rome
臂 throttle 从 29+25 提到 50+40（用了那个看着像权限拒绝但实际生效的 `ArrayTaskThrottle`
更新，集群反复出现的怪癖）。

**homo SP 真实 ETA 比承诺的差很多。** 没信 09-10 那个"~2 天"的估计，直接测实际吞吐：
archived 轨 ~45 shard/h，regen 轨 ~4.1 shard/h —— 都外推到 **~7-7.5 天**，不是 2 天，
因为昨晚的 QOS 公平性修正把并发从计划的 ~250 砍到了 genoa 单区 72。诊断了*为什么*卡在
那：我们 genoa 上的占用（homo_sp 60 + homo_regen 12）加上 NHC 自己的 `nhc-gsp-*`
作业（~54）已经顶在共享的每分区 128 QOS 上限——genoa 加不动了，fat_genoa 也满。跟
Tier B 一样的修法：rome 有真实空闲、NHC 也基本不在那，加了两条 rome 补给臂
（`26573745` archived 3000-8971%25、`26573746` regen 300-770%6，起始编号避开 genoa
已派发的低段）。加完后预计：archived ~5.4 天 / regen ~4.8 天（≈09-16/17）。下一个杠杆
（还没拉，等 drain 通知再动，免得撞车）：Tier B drain 后把它腾出的 ~90 个 rome 名额大部
分转给 homo_sp archived，ETA 有望压到 drain 后 +2-3 天。

**两条战役都在等 compute 时，推进 flying dataset（CHEMICAL_SPACE.md §8）第 1-2 步。**
正是 PROJECT_PLAN §6 第 3 项标的"无计算、慢慢理解"的下一步。第 1 步：冻结了
`data/chemical_space/aldehyde_index.parquet`（220,859 行——修正了两份计划文档里传播
的一个 off-by-one，CSV 是 220,860 *行*含表头，不是 220,860 个醛）。发现并复用（不重算）
了 `candidates_v3/aldehydes_with_scaffold_split.parquet` 里给这同一批醛已经算好的
Bemis-Murcko scaffold（`pipeline/bde/build_scaffold_splits.py` 给 BDE 项目复用的同一个
文件）——先在**全量** 220,524 行重叠上校验了它的 `id` 列逐位 == `ald_idx`（0 处不符，
raw SMILES 精确匹配），不是抽样，才信这个合并。第 2 步结果是校验而不是重建：
`homo_v6/aldehydes_all.csv` 现有的 `id`（经 `qc.norm_id`）在全量 209,526 行上已经
1:1 等于 `ald_idx`，两边 0 孤儿——早就 key 对了，只是没写文档。路上真发现一个问题：
那个文件的 `smiles` 列不可靠地 canonical（1.24% 跟新算的 canonical 形式只是表示法不同，
比如 Kekulized vs 小写芳香）——以后的缓存 join 要用 `ald_idx`，不能用 SMILES 字符串
相等，这正是冻结索引要消除的漂移。parquet 用 `git add -f` 提交（默认被 gitignore，跟
项目里其它"不能再丢一次"的产物一个纪律）。

--- 快照（2026-09-11）：两条计算战役都还在跑（Tier B ~78%，homo SP 慢但已加速）；
flying dataset 构建顺序到第 2/6 步；PROJECT_PLAN + CHEMICAL_SPACE（+ZH）已同步更新。
下一个设计步骤：第 3 步—— `chemical_space.py` 的 `pair(i,j)` 特征路径，必须对 ~20
个已知对做校验。 ---

**同一晚，接着做（当时没记）：** 第 3 步完成——`FlyingDataset.pair(i,j)`，两条确定性
lazy tier（RDKit-2D、`interaction_*`、`product_smiles`）对 round-10 表 20 个随机对做到
逐位精确匹配；donor/acceptor QM 72% 在 0.2% 内匹配（其余是已知的 aldehyde-recompute-
fidelity 效应，不是 bug）。抓到一个真 bug：早期校验版本用 `pair_key` 查找来定位一行的
(donor, acceptor) 地址，当一个 `pair_key` 对应两种角色顺序时会悄悄选错行——改成直接
从该行自己的 donor_smiles/acceptor_smiles 解析。接着做了个消融回答用户"为什么是 260
个特征"的问题：去掉 91 个产物侧 QM/mordred 列（产物几何不是 lazy 的，CHEMICAL_SPACE.md
§5）代价 **+1.00 kcal MAE**（2.544→3.548，+39.5%，5-seed 合并 sd 仅 ~0.02——真实、有
统计力）。仍远好于 g-xTB 基线（5.037），所以 lazy 子集是合理的廉价初筛，但不能替代
完整流水线——flying dataset 不可能做到完全 lazy 而不付精度代价。

### 2026-09-12 / 09-13 —— 无会话

两天都没有提交。

### 2026-09-14 —— Tier B drain：b973c 冠军突破并上线（2026-09-15 事后重建）

*当天没记日志；2026-09-15 恢复会话时从提交记录（`50d37b7`、`3a894bc`、`273d74f`、
`e798755`）和 CHAMPION.md 内容重建——标记为事后重建，不是实时记录。*

**homo 醛化学型聚类诊断**（为 flying-dataset 切分设计做准备）：对全量 220,859 个醛的
ECFP4 指纹做 MiniBatchKMeans（k=150），关联当时 43.0% 覆盖率的 homo SP 在飞标签，在
3 种切分方式 × 4 种方法下评估。发现：random CV 下看到的 R²~0.03 聚类/scaffold 组均值
信号是**泄漏假象**——在 cluster-disjoint 和 scaffold-disjoint GroupKFold 下都塌缩到
~0，而原始指纹结构在三种切分下都保留一点真实信号（R²~0.04-0.05）。证实 flying-dataset
切分必须按结构分组，且 2D 化学型本身解释不了 homo 目标的多少方差（与
homo-active-relabel-null-result 的转向结论一致）。中途基础设施修复：一个基于 PCA 的
结构分支卡住了（撞到 1 小时 SLURM 限制），原因是这个 venv 的 numpy 在这代 CPU 上
BLAS 矩阵乘法跑的是未向量化路径（76000×2048 矩阵乘法实测 14 秒，预期 <1 秒）——改用
纯 numpy 的 ECFP4 位折叠（2048→128，数据无关，无跨折泄漏风险），几分钟内跑完。以
真正执行过的 notebook 交付（`notebooks/cross_benzoin/homo_aldehyde_cluster_dG_
analysis.ipynb`），是 2026-09-14 用户"分析用 notebook 格式交付"偏好下的第一个。

**Tier B（Rec-1 廉价基线杠杆）drain 完成**，覆盖率 98.9%（35,136/35,528），QC 绿灯
（B97-3c 残差标准差 0.916 vs g-xTB 的 3.785，比值 0.242——与全量 35k 规模下 pilot 的
0.26 吻合）。按 `DRAIN_RUNBOOK.md` 步骤 1-6：在 B97-3c 基线的 Δ 目标上重训冠军流水线
（schema v2，257 特征）。结果（同一冻结 n=448 holdout）：单 XGB 0.632，MLP+XGB
ensemble 0.603，GNN 4-seed 平均 0.531，**blend（w_gnn=0.85）0.528**——对比 g-xTB 冠军
的 2.215，**降了 76%**。远超 2026-09-10 PROJECT_PLAN 猜测的"~1.0-1.5 = 突破"门槛——
真实收益比猜测的大得多。机制：Δ 模型现在只需要学 `r2SCAN-3c − B97-3c`，这个残差在任何
机器学习之前就已经很紧（std 0.916），因为 B97-3c 起点比半经验 g-xTB 离标签近得多。

**同一 session 接入 `predict_dg.py`**（`cb_featurize.py --with-b973c` 给全新对算新基
线）。端到端测试时发现两个早于本 session 就存在的 bug，同时挡住 g-xTB 和 b973c 的
from-scratch 路径：(1) `submit_predict_dg.sh` 读一个 `cb_featurize.py` 从未写过的
`features.csv`（实际写的是 `products.csv`）；(2) 2026-09-08 的 n_CHO 特征审计早已悄悄
弄坏了对任何新对组装表格去匹配仍是 260 特征的已部署 schema（`calc_rdkit` 仍算 n_CHO，
只是组装器的列*选择*丢了它）——用 `include_ncho=True` 修复。

**真实对精度检查**（全新 GFN2-xTB + ORCA 从头算，不是训练表回放，走真实的
`submit_predict_dg.sh` → `predict_dg.py` 流水线）：2 个干净的 test-split 对上，b973c
MAE 0.34 vs g-xTB MAE 2.50——都恰好落在各自的 holdout MAE 上，独立证实 b973c 的收益
不是训练表的人工产物。第 3 个对（一个容易两性离子化、类氨基酸的供体）**两个模型都
失败**——是上游共享的特征/几何问题，不是 b973c 特有的；g-xTB 路径的 `dg_high_sigma`
正确抓住了它。**CHAMPION.md 翻转**：r1-10-b973c 现为冠军+默认部署，r1-10（g-xTB）降
为文档化 fallback。session 结束时标记的已知缺口：还没给 b973c 建 calibration 产物。

### 2026-09-15 —— 断档后恢复：b973c calibration + 一个真实的 seed 平均部署 bug

**冷启动恢复**（09-11 尾巴之后没有 HANDOFF 或日志条目）。先从 `git log` + 提交正文
重建了 09-12/13（无会话）和 09-14（密集但未记日志）——见上两条。确认 homo SP 战役
仍健康：archived 6268/8972 shard（69.9%），regen 654/771（84.8%），合并吞吐 ~50
shard/h（archived 是瓶颈）→ **ETA ~2.3 天（≈09-17/18）**，接近 09-11 的预测。

**接手 CHAMPION.md 标记的缺口**：通过泛化 `build_predict_dg_calibration.py`（原来
硬编码 g-xTB 冠军的表/模型路径；现在接受 `--table --model-dir --gnn-dir
--blend-w-gnn --baseline-col --label-col`，先验证对原 g-xTB 配置逐字节等价（除
1e-13 级浮点噪音）才敢信这个重构）构建了
`cross_benzoin/predict_dg_calibration_b973c.json`。

**过程中发现一个真 bug，不是表面问题**：按 CHAMPION.md 文档的"Load"方式
（`CrossBenzoinBlendPredictor.load(..., gnn_dir=seed4)`）算出的 b973c holdout MAE 是
**0.544**，不是冠军文档写的 **0.528**。根因：0.528 是 4-seed GNN 平均后的数字
（`gnn_seed_ensemble_r10_b973c_result.json` 的 sweep，`w_gnn=0.85`），但 2026-09-14
session 接入 `predict_dg.py` 时只加载了单个 seed 目录（seed4 单独，`w_gnn=0.70`，来自
该 seed 自己的 `metadata.json`）——CHAMPION.md 自己的"Load"示例代码内部就不自洽
（声称 w_gnn=0.85"来自 GNN metadata.json"，却指向一个 metadata 实际写着 0.70 的单一
目录）。跟 09-08 g-xTB seed-ensemble 那次发现（2.215→2.137）是同一类缺口——不同的是
那次 09-08 session 正确判断收益还不显著、没有采纳（留作文档历史，这次没动）；这次
09-14 session 已经决定把 4-seed 数字当冠军了，只是代码没跟上。

**彻底修复，不是绕过去**：`CrossBenzoinBlendPredictor.load()` 现在接受 `gnn_dir` 的
*列表*，用每个成员自己的反归一化统计量分别预测再平均（验证了各 seed 自己的
`ym`/`ysd` 确实略有不同，~0.003-0.009，所以必须按成员分开算，不能共用一套统计量）；
传列表时 `blend_w_gnn` 必须显式给出（每个 seed 自己的 metadata.json 只调过它自己单
独的权重——瞎猜该用哪行 sweep 结果会悄悄上线错的权重，跟本文件一贯的"明确报错，不
悄悄出错"原则一致）。重新验证：4-seed 加载复现 MAE 0.5284，与 sweep 结果对到小数点
后 5 位。`predict_dg.py` 的 `--gnn-dir` 现支持逗号分隔列表 + `--blend-w-gnn`；
`build_predict_dg_calibration.py` 同步支持。用*正确*的 4-seed 冠军重新构建了
calibration 产物（第一次用错误的单 seed 配置构建的版本，在上线前就被发现并重做了）：
split-conformal 90% 区间半宽 **±1.21 kcal**（vs g-xTB 的 ±5.24——收紧 4.3 倍，与 MAE
差距吻合），覆盖率 test 0.908 / validation 0.894。

**还有一个真发现，不只是搭线**：那 7 个能清楚标出 g-xTB 基线失败的 SMARTS 官能团
（标记行的|基线误差| 6.35 vs 未标记的 4.81）对 b973c **不代表同一件事**——标记行的
blend MAE 更高（0.80 vs 0.50），但 |基线误差|**持平**（4.94 vs 4.98）。B97-3c 本身在
这些结构上并没有失败；是 Δ 模型觉得这些结构本身更难。已在 CHAMPION.md 和
predict_dg.py 自己打印的摘要里都写清楚，避免调用者把 b973c 路径的
`baseline_risk=True` 也误读成"改用 DFT"（这个读法对 g-xTB 仍然是对的）。

`predict_dg.py`/`predict_cross_champion.py`/`build_predict_dg_calibration.py`/
`CHAMPION.md`/`PROJECT_PLAN.md` 一起更新；在当作完成之前，把整条流水线（加载 4-seed
冠军 → 预测 → 接 calibration 列 → 官能团标记）在 20 行真实 holdout 数据上原地重新
跑过一遍，不只是"能 import 就行"。

**同一天，接着做：flying dataset 构建顺序第 4 步。** homo SP 是唯一还在等计算的战役
（~2.3 天），于是接手了排队的设计任务（PROJECT_PLAN §6 第 2 项）。`FlyingDataset.pair()`
现在直接返回 cache-hit tier 的 `label` / `label_col` / `split` /
`baseline_gxtb_kcal` / `baseline_b973c_kcal`（不在 `known_pairs` 里的地址老实返回
`None`）。真正的设计问题是列名解析：g-xTB 时代冠军表的标签叫 `dG_orca_kcal`，根本没有
`dG_b973c_kcal` 列；现在的冠军——b973c Tier B 表——真实标签是 `dG_r2scan_kcal`，两个基线
都有。固定一个列名会在没测过的那张表上悄悄失效。`LABEL_COL_CANDIDATES`/
`SPLIT_COL_CANDIDATES`/`BASELINE_COLS` 给每个字段按顺序试一串候选名，模块不改代码就能
对两代表都用。给 `verify_chemical_space_pair.py` 加了 `step4` 层（在测试里独立解析，
不是重新调用模块自己的逻辑——不然解析顺序的 bug 会自我印证）和 `--table` 参数，然后
对**两张表都跑了**（不只是默认那张）：每张表 20/20 对通过，80/80 个 step4 字段精确
匹配，`label_col` 在一张表上正确报 `dG_orca_kcal`、在另一张上正确报 `dG_r2scan_kcal`。
正是这个跨表验证才真正测到了解析逻辑——只测默认表的话，"两张表用同一个列名"这类 bug
根本测不出来。

**同一天，接着做：第 5 步，用户选了"宽做"。** 先问了用户还有哪些事需要他决断；给第
5 步列了窄做/宽做两个选项（只接读取 API vs. 真的把 candidates_v3 的文件和它的 13 个
依赖脚本都退役）。用户选了宽做——但先反问了一个更尖锐的问题，把整个任务重新定了框：
"candidates_v3 跟你有什么关系？你用的不是v6版本的醛数据库吗"。要答好这个问题，必须
真去查那个目录里到底装着什么，而不是照字面信"retire candidates_v3"这句话——一查，
发现用户的直觉是对的：`candidates_v3/aldehydes_with_scaffold_split.parquet`
（10MB，`build_aldehyde_index.py` 和 `pipeline/bde/build_scaffold_splits.py` 都还在
实际读它）是一个 v6 醛级别的资产，从来不属于要退役的那个配对池——只是碰巧放在了
"candidates_v3"这个路径下，这正是那个问题戳中的混淆点。如果盲目"整个目录搬走"，会
同时弄坏一个活跃的 BDE 流水线读取和 flying dataset 自己的索引构建脚本。

**动手前先做了完整的依赖审计**：grep 了全部 26 个引用"candidates_v3"的文件，分成三
类——(1) scaffold parquet 的活跃读者（2 个文件，真有风险），(2) 本项目自己批评的
那个"~124 万对任意子集"（CHEMICAL_SPACE.md §1），被 13 个采样/训练/分析脚本引用，
(3) 纯文字历史注释（无代码风险）。查第 2 类时发现了意外情况：它引用的那两个配对池
文件（`candidates_v3_pairs_with_scaffold_split.parquet`、`inchikey_split_map.parquet`）
在磁盘上根本不存在，现存的两个 `.csv.gz` 文件（`cross_benzoin_dG_candidates_v3.csv.gz`、
`cross_benzoin_aldehydes_v3.csv.gz`）只有 133-134 字节，不是有效的 gzip——这批数据早
在 2026-07 的 purge 里就丢了，从没恢复过（跟 cross AL 被 PARKED 的现状吻合）。所以
第 2 类脚本在这次搬迁之前就已经跑不通了；退役它们是清理和路径卫生，不是弄坏一个活的
依赖。确认了实际部署的重训脚本（`train_scaffold_disjoint.py`，DRAIN_RUNBOOK.md）
在 import 时只从 `train_cross_delta.py` 拿两个环境变量可覆盖的常量——运行时不读任何
candidates_v3 文件，它自己的切分逻辑用的是训练表自己的 `new_scaffold_split` 列（这
正是 `train_scaffold_disjoint.py` 存在的原因——它取代了那个会泄漏的
candidates_v3-切分训练）。

**执行**：`aldehydes_with_scaffold_split.parquet` 搬到 `data/library/`（跟
`aldehydes_clean_v6.csv` 放一起，它本来就该在那）；它的 2 个活跃读者 + 1 个写者的
路径都改了，通过解析路径常量 + 重跑 `verify_chemical_space_pair.py` 验证过（两次
跑分别还是 10/10、15/15）。目录里剩下的东西（README、manifest、QA xlsx、
representativeness_check/、两个已经死掉的 .csv.gz 残骸）都归档到了
`data/cross_benzoin/_archive/candidates_v3/`，附一份 `RETIRED.md` 说明搬了什么、
本来就丢了什么、为什么。13 个依赖脚本的路径都更新到了归档位置（这样即使大多数早就
跑不通了，也还能保持可复现/可 grep）；5 个一次性 AL 轮次采样脚本 + representativeness
分析脚本加了退役说明，指向 flying dataset 作为替代。`train_cross_delta.py` 的
`SPLIT_MAP` 加了行内注释说明它是遗留的（被骨架不相交切分取代，不在当前冠军链路里），
而不是悄悄改个路径不给解释。

**顺带把第 5 步的另一半也做了**（"迁移标签"，不只是"retire candidates_v3"——构建
顺序原文把两件事写在同一行）：写了 `build_labeled_pairs.py`，冻结出
`data/chemical_space/labeled_pairs.parquet`——35,136 条已用标签的对（来自 b973c
Tier B 表，现在最全的那张），通过 **InChIKey**（不是 `chemical_space.py` 自己那套
对任意表用的 SMILES 归一化查找——这个脚本控制自己的源列，可以直接用精确 key，绕开
那整类歧义）关联到 `aldehyde_index.parquet`：35,136/35,136 全部解析成功，0 个地址
冲突。给 `FlyingDataset._build_known_lookup` 加了一条快速路径：如果 `known_pairs`
表直接带 `donor_ald_idx`/`acceptor_ald_idx` 列（这张新标准表就带），就完全跳过 SMILES
往返。端到端验证过：新表里随机 100 行，走新路径在 label/split/两个基线上全部精确
匹配；旧的 SMILES 路径事后对两张冠军表重跑也干净（没有退化）。
`LABEL_COL_CANDIDATES`/`SPLIT_COL_CANDIDATES`/`BASELINE_COLS` 也扩展了，让标准表
自己的列名（`label`、`split`……）排在最前面优先命中。

--- 快照（2026-09-15，日终）：homo SP 是唯一在跑的计算，ETA ~09-17/18，除了等待和
监控没有别的事。cross-benzoin 部署这条线目前没有已知缺口。flying dataset 构建顺序
现在 5/6：第 1-5 步完成，第 6 步（把重做的 cross AL / Goal-3 筛选接上去）依赖的是
采集策略的重新设计，那是要用户拍板的设计选择，不是 flying dataset 本身还欠工程量。
这次会话里抛给用户、还在等他决断的：催化剂空间/NHC 仓库整合范围、Rec-2 的 −0.11
发现在 b973c 地板变了之后还值不值得重测、以及重做 AL 的采集策略本身。 ---

## 2026-09-16

**只做了状态检查，没改代码。** 这次是用户要求汇报进度而恢复的会话；现在能做的事
要么卡在 homo SP 计算 drain 上，要么卡在 09-15 已经抛给用户的三个决策上（催化剂
空间范围、Rec-2 重测、AL 重新设计），所以只做了健康检查，没有新工作——为了显得
"在干活"硬凑填充任务，比如实说"还在等"更糟。

相对 09-15 09:41 快照的进展：archived 轨道 6268/8972 → **7350/8972 分片
（81.9%）**；regen 轨道 654/771 → **771/771（已完成，766/771 已标记 `.done`，
剩 5 个正在收尾）**。合并吞吐量与 09-10/09-11 已经诊断出的 ~50 分片/小时
archived 侧瓶颈一致——不需要新的限流调整。检查了所有 6 个作业数组 ID（genoa +
rome 补量臂 + fat_rome ×2，archived 和 regen 各一套）的 `sacct`：20,825 个
COMPLETED / 159 个 RUNNING / 27 个 PENDING，只有 3 个 CANCELLED+（可忽略，符合
正常的 requeue）——没有故障模式值得追。鉴于一直存在的 [[scratch-disk-quota-risk]]
担忧，检查了配额：home 61.8%/67.9%（GiB/inode），scratch1 16.2%/31.6%——健康，
现在没有"最后写入步骤悄悄丢结果"的风险。剩余 archived 分片（1622 个）按 ~50/小时
算，大约还要 **32 小时，仍然落在 09-17/18 的 ETA 里**，不需要修改。

还没跑的：drain 流程（`merge_homo_sp.py` → QC → `--full-library` assembler →
单个 XGB + 单个 GNN，PROJECT_PLAN §6 第 1 项）——正确地卡在 archived 轨道真正
到 100%，而不是接近就跑。分片跑完后会按 [[handoff-routine-and-autonomy]] 里
既定的自主推进授权直接执行，不用再等一次提示。

**同一天，继续：用户要求给 dG 的 GNN 那条腿尝试 chemprop 和反应图（CRG）**，
同时顺手用已有的部分 homo 表跑了一版 Rec-2 早期读数。

**Rec-2 早期读数（临时，homo SP 还没跑完）**：重跑 `assemble_homo_standalone_
table.py --full-library`（136,874 行部分 merge_homo_sp.py 输出）之前先发现并
修了一个真 bug：脚本从没 join 过醛侧 mordred（`donor_ald_mordred_*`/
`acceptor_ald_mordred_*`，257 个冠军列里的 67 个），从写出来那天起 30k 和全库
两条路径都在用一个残缺 schema 训练；脚本自己文档里"champion has none"的说法
在 round8 时就已经是错的。修好后（join `aldehydes_mordred_slim102.csv`）。
写了 `homo_cross_joint_tabular_v2.py`（Task C 方法论搬到 257-feat+b973c 尺度，
加了 `naive_merge_weighted` 条件直接测 FINDING.md 担心的"6:1 会稀释信号"）。
在 116,740 行临时表（homo:cross 5.18:1）上的结果：naive_merge（不降权）比
cross_only 好 0.046（0.617→0.572），AMBER（n=448 下大概率不到 1 个 bootstrap
SE）——而且"不降权"反而比"降权"版本更好，和稀释假说预期的方向相反。只是
临时读数，等完整表落地才能定论。

**chemprop**：发现项目共享的 chemprop 环境（`envs/gnn`、`envs/bde_gnn`，在
`/gpfs/scratch1/shared/schen3/envs/` 下）全部在 purge 后损坏（bin/ 目录空的），
但 `/home/schen3/venv/bde_gnn`（home 目录，~09-02 重建）能用（torch 2.13+cu130,
chemprop 2.2.0；补装了缺的 pyarrow）。写了 `train_cross_gnn_chemprop.py`，用
chemprop v2 原生的 `MulticomponentMessagePassing`/`MulticomponentMPNN`（产物/
供体/受体各一个独立 `BondMessagePassing`，`shared=False` 对应 TripleGNN 三个
独立编码器的设计）+ 257-feat schema 当 `x_d`。CPU smoke test 后跑了一次真实
GPU（gpu_a100，21 分钟）：单 seed test MAE **0.600**（n=448）——打平 MLP+XGB
ensemble（0.603），比单 XGB 好（0.632），不如 TripleGNN 单 seed（0.556）。没
做 seed 集成（chemprop 是次要需求，CRG 才是重点跟进对象）。

**写 chemprop 脚本时顺带发现一个真 bug，没在那修**：`train_cross_gnn.py`
自己的 split 逻辑（`train_cross_delta.pair_split_labels()`）读的是 candidates_v3
的 `SPLIT_MAP`，09-15 已经退役——文件不存在，函数现在无条件返回 `None`，下游
代码把这个变成"每一行都扔进 train_extra"（val/test 变空）。现有的冠军 GNN
checkpoint 都是退役之前训的，不受影响；只有"下一次重跑"才会踩到这个坑。当天
晚些时候修了（见下）。

**CRG（反应图）——真正的重头戏。** 关键洞察，让这件事不需要外部反应原子映射
工具：benzoin 偶联（2 RCHO → R-CO-CH(OH)-R'）是原子经济反应，产物 SMILES 本
身就已经完整包含供体+受体的全部原子（作为完整的取代基树）——这里的 CRG 不需要
把三个 mol 对象拼一起，只需要（a）用一条 SMARTS（`[CX3](=O)[CX4][OX2H1]`）在
产物图里定位 5 原子反应核心（ketC/ketO/carbC/hydO，没有单独的 hydH 节点，因为
这个项目的图是隐式 H），两次独立抽样（500/2000 行）验证唯一匹配率 97.5-97.6%；
（b）纯靠产物图自身连通性，用 BFS 把其余原子按落在新 ketC-carbC 键的哪一侧
分类——完全不需要交叉参照 donor_smiles/acceptor_smiles。形式上的净原子映射
（质量守恒自洽，不是对 NHC Umpolung 真实中间体的机理断言）写在 `crg_builder.py`
文档里。`crg_builder.py` 纯 RDKit 实现，先在一个玩具分子上单测过（side/core
标签和新键 edge flag 全部验证正确）才碰真实数据。

`train_cross_gnn_crg.py`：单个 `Enc()`（和 TripleGNN 每个分支用的同一个
GINEConv block，这样差异只来自"连通 vs 不连通"的表征本身，不是特征集变化）
处理这一张condensed产物图 + 257-feat 的 x_d 通道。split 和 chemprop 脚本一样
用 `new_scaffold_split`（同样是为了绕开刚发现的那个 landmine）。

**第一次单次跑：0.559**（n=432，约 2.5% 的行因反应核心 SMARTS 没匹配/歧义被
丢弃）——先做了公平性检查才敢兴奋：冠军 blend 在同样 432 行子集上的 MAE 是
0.5295，全量 448 行是 0.5284，子集不是偷偷变简单了，对比是站得住的。接着在
"seed 1/2/3"续跑上真的判断错了一次：这个集群的 `SEED=$s sbatch ...` 不会把
环境变量传进作业（sbatch 的 site 默认 `--export` 不像 `SEED=$s sbatch` 字面
看起来那样继承 shell 变量），三个"不同 seed"其实悄悄全跑了 seed 0——发现的
线索是"报出来的 seed0 数值每次检查都不一样"（CUDA 在固定 seed 下也不是完全
确定性的）。用户两次追问（"seed是否太少"/"还是数据太少"）都问对了：那 n=4 的
四个数（0.559/0.564/0.604/0.593）看起来像个宽、让人担心的分布，但 n=4 根本
分不清是噪声还是真的双峰失败模式。

**统计功效够了的版本**：修好 sbatch 的 bug（`--export=ALL,SEED=$s`），真正
跑了 30 个独立 seed。均值 0.572 ± 0.012（min 0.554, max 0.605）——之前那个
"0.610 离群点"本身就是小样本的幻觉，不是真正的双峰尾巴。加了真正的预测级
集成（逐行 y_pred 先跨 seed 平均，不是平均 MAE 数值——和冠军自己的 4-seed GNN
数字算法一致），靠一次脚本改动存下 `test_predictions.csv`：4-seed 集成 0.543，
之后就饱和了（8/16/30-seed：0.543/0.542/0.543）——和冠军自己 seed 集成"4 个
左右饱和"的规律一样。

**超参搜索**（用户接着要求"进行超参搜索"）：CRG 之前全程用的是 TripleGNN 的
未调超参。20 组随机配置（hidden/layers/lr/dropout/weight_decay）× 2 seed =
39/40 个作业（1 个因集群瞬时"compute budget"错误没提交上），只按**验证集**
MAE 选型（test 全程没碰）。最优：hidden=128 layers=4 lr=3e-3 dropout=0
wd=1e-4——和默认值很接近，主要是学习率高 3 倍、去掉了 dropout。用这个配置
再跑 12 个 seed：12-seed 集成 test MAE **0.535**（n=432），基本打平冠军在
同一子集上的 0.5295。

**用户第三次追问**（"n=432 是否太少了 需要更多的数据"）——又问对了：把评估
扩到 pooled test+validation（n=898，和 `build_predict_dg_calibration.py` 已经
在用的惯例一样），这需要 CRG 也在 validation 集上跑推理（写了个小脚本复用
训练脚本自己的 `build_crg`/`CRGGNN`/`make_loader`，用确定性的 train 行集合
重新算出同一套 leakage-safe x_d 标准化统计量，不需要提前存过）。合并结果：
冠军 0.5556，CRG（调优后，12-seed）0.5635。bootstrap（20000 次重采样）：
CRG-冠军 差值均值 +0.0079，90% CI **[-0.0029, 0.0189]**（P(CRG 更差)=88.6%，
比 n=432 时的 75% 更高）——数据变多后信号从"分不清"变成"冠军大概率还是略好
一点"，但 90% 区间仍然勉强包含 0。把之前"基本打平"的说法修正为"接近，大概率
略逊一筹，差距不大"。还发现一个没解释的现象：validation 子集对两个模型都比
test 子集更难（冠军 0.58 vs 0.53，CRG 也是 0.59 vs 0.53），而且 CRG 相对冠军
的差距在 validation 上略微拉大——记了一笔，没深究。

**试了把 CRG 和 tabular ensemble 做 blend**（冠军自己的配方：`(1-w)*ens_delta
+ w*gnn_delta`，w 只在 validation 上选）。结果：w=0.76，test MAE 0.5352——
统计上和 CRG 单独跑（0.5348）没差别，没有 blend 增益。事后想想合理：CRG 自己
的 x_d 通道已经融合了同一套 257-feat schema，一个只训练在同样特征上的 tabular
模型加不了 GNN 已经看到的信息——不像 TripleGNN，从 tabular 堆叠里明显受益
（w_gnn=0.85，不是 1.0）。是个真实的负结果，不是 bug。

**修了上面发现的 `train_cross_gnn.py` landmine**：当表里有 `new_scaffold_split`
列时（现在的表都有）直接用它（`mixed` → `train_extra`，保留脚本原有的
train/train_extra/validation/test 四桶语义），只有表里没这列时才退回旧的
candidates_v3 路径。对着真实的 b973c 表验证过：复现出了完全一样的已知 split
行数（22529/11678/481/448）。

**目前的状态**：CRG 是个真实、验证过、能跑通的架构——接近但（现在统计功效
比较够了之后看）大概率还是比调过的成熟 TripleGNN 差一点，不是明确的胜利。
没有下结论"采用"或"关闭这条线"——如果继续推进，开放的线头：（1）CRG 自身
的架构改进（`crg_builder.py` 文档里提到但没实现的"键级变化"edge flag；把
CRG 和 TripleGNN 本身而不是 tabular ensemble 做 blend）；（2）同样的 CRG
思路用在 BDE 上（用户和 dG 那个请求一起提的，"BDE等工作也可以尝试"——这次
没开始，单分子 + 显式标记目标键是自然的类比，但需要单独搭建）。

**同一天，会话在做 BDE-CRG 那段被中断，下次会话冷启动接手。** 上面那段写完后
会话确实继续搭了 BDE 版本（`pipeline/bde/train_gnn_hybrid_bde_crg.py`，只做
产物侧 Task B，醛侧 formyl C–H 在本项目隐式加氢表示下没有可标记的图边），
也提交了全量消融（5 seed × marked/unmarked，gpu_a100），但会话在看结果之前
就结束了，没人看、没记录。

**恢复后发现：全量消融跑崩了，不是"效果不好"那种崩，是真的没训出东西。**
已完成的 10 个 job 全部异常：MAE 在 9~239 kcal/mol 乱跳，R² 普遍 ≤0（最差
-7.29）——**连理论上该是无操作的 `--no-mark` 对照组也崩**，而同一套底层训练
代码在 champion B6 产物任务上单 seed 稳定给出 MAE 2.07–3.24（从未改动过的
`b6_ensemble_26418250_*` 日志核实过）。更巧的是，恢复时发现第二批一模一样的
10 个 job（26799798-809）还在跑，正在原样重复这个已知会崩的实验。

**用 n=5000/15-epoch 的 CPU 小规模三方对照排查**（baseline 原始脚本 / CRG
no-mark / CRG marked，同 seed 同 split 同超参）：三个全部训练正常
（R²~0.80-0.81，MAE 4.2-4.8），排除了 `extra_bond_fdim`/`E_f` 特征注入机制
本身有 bug（直接核对了 chemprop 的 featurizer 源码和 bond index 对齐，不是
memory 里记过的那类 ALFABET 对齐 bug）。**目前最可能的解释**：`train_one`
用 `enable_checkpointing=False` + EarlyStopping 不 restore best weights，
champion 脚本也有这个弱点，但全量数据 + 最多 120 epoch + 更大模型
（d_h=500）给了足够多步数让权重在 patience=20 触发前跑偏——15-epoch 小样本
测试摸不到这种失效模式。**没有证实**（两批全量跑都没开 logger，没留逐 epoch
记录），**没有修复**（碰 `train_one` 会影响 champion 的可复现性，没经用户
确认不单方面改）。想 `scancel` 那 10 个重复的 job，被 auto-mode 的工作负载
保护拦下了，已转交用户决定，任务本身没杀。

**顺带查了用户问的备份/同步问题（"是否有效备份和同步关键数据"）**，三层：
（1）git/GitHub：7 个本地 commit（一路到今天的 CRG 工作）一直没推送，
上次推送是 09-14——已推送。（2）账号上唯一名字里带"benzoin"的自动备份脚本
`backup_benzoin_scratch_weekly.sh` 实际指向的是 `/scratch-shared/schen3/workfow`
（**NHC 催化剂项目**的目录，不是这个 benzoin-dg-restored 仓库），而且从
09-08 起就卡死了——一个没有活跃进程持有的 `.backup.lock` 让每次重试都直接
打印"backup already running"退出，某个每 20-30 秒触发一次的东西（没找到是
cron 还是别的）已经这样空转了一周多，往 `logs/` 里塞了上万个小日志文件。
**净效果：这个项目在 scratch1 上的数据已经两周多没有自动备份覆盖**，而且
名义上存在的"备份"其实完全是另一个不相关项目的。这个不是本项目的脚本，
没有动它，留给用户处理。（3）BDE 自己 STATUS.md §7 写明要 `git add -f` 的
`aldehydes_all.csv`（100MB）和 `products_all.csv`（153MB）实际上根本没有
执行——两个都被 `.gitignore` 挡住，从未 force-add，也没有 home 备份，只存在
于 scratch1 一份，和当年造成 purge 丢失的情形一模一样。checkpoint（`.pt`）
倒是确认有正常入库。**已修复**：两个文件都 rsync 到
`/home/schen3/benzoin_backups/bde_critical_data_20260916/`；
`aldehydes_all.csv`（95.5MB，在 GitHub 100MB 硬限以内）额外 `git add -f`
推送（`e8d3656`）；`products_all.csv`（153MB，超限）目前只有 home 备份一份，
上 GitHub 需要 git-lfs，还没配。

## 2026-09-17

从 `HANDOFF_20260916.md` 续接。按其 §1.2 第一优先级,查了
`train_one` checkpoint-restore 修复的全量验证结果。

**结果**:`26807944`(marked,CRG 标记目标键)干净跑完,用时 3:48:14——
MAE 3.140,RMSE 5.338,R² 0.886,spearman 0.945(170,996 行 scaffold-disjoint
产物集,138268/15363/17365 train/val/test)。正好落在预期的 2-4 kcal/mol /
R² 0.85-0.9 区间内(对照 STATUS.md §2.1b 的 B6 单 seed 2.09/3.19)——
**确认 checkpoint-restore 修复解决了训练不稳定问题**,不是数据或架构的锅。

`26807945`(unmarked `--no-mark` 对照组)撞了 4 小时墙钟超时没跑完——值得
注意的是它没有像 marked 那样提前收敛,暗示 CRG 目标键标记可能加快收敛
(这只是观察,要等 unmarked 跑完才能下结论)。已用 `--time=08:00:00` 重新
提交,并用正确的 `sbatch --export=ALL,SEED=0,MARK=0 ...` 写法(job
`26829622`)——确认这次变量正确传进去了,没有重复 09-16 踩过的
`SEED=0 sbatch` 环境变量不传的坑。后台挂了一个等待循环,跑完会通知。
`pipeline/bde/STATUS.md` §9 已写完整表格,§8 标题已改指向它。

homo SP archived 分支续接时是 8488/8972(94.6%),比 09-16 交接时的
7841/8972(87.4%)又推进了;regen 分支已完成 771/771。后台挂了一个轮询,
archived 到 8972/8972 就提示,这样 Phase 2 合并/组表 + Phase 3
单 XGB/单 GNN(CAMPAIGN_PLAN.md)可以及时启动,不用会话一直盯着。

留意但没动:09-16 交接 §4 标为"来源不明"的 6 个 `cross_round9` 文件改动,
确认是**真实的一次重训**,不是统计噪声——`gnn_attentive_9rounds_v1/models/
metadata.json` 显示 n_train 39030→18728、n_val 4820→2565(近乎腰斩),
MAE 2.16→2.31,best_blend_w_gnn 0.55→0.40。有什么东西用不同(更小)的划分
重跑了 round9 的训练。来源仍未查清;round9 已被 round10 取代,风险不高,
按交接文档自己的建议("不确定就先别动")继续不提交、不动,等想起是谁跑的
再说。

**用户要求尝试其他基于反应的GNN架构**("尝试其他的基于反应的GNN")，经简单
澄清后确定范围为 cross-benzoin dG。在已有的 champion TripleGNN(拼接、3个独立
编码器)和纯CRG(图合并、单编码器)基础上新增两个对比点:

1. **CGR-delta**(`crg_builder.build_crg_delta()` + `train_cross_gnn_cgr_delta.py`):
   纯CRG自己文档标注过但没做的改进——把单一的is_new_bond标记升级为真正的
   "动态键"前后键级编码。反应中实际有两条边键级会变(不只一条):ketC-carbC
   (新建,0→1)和carbC-hydO(键级改变,受体的CHO C=O变成产物的C-OH,2→1)——
   第二条之前对模型完全不可见。边特征从7维升到12维。
2. **WLDN风格差分网络**(`train_cross_gnn_wldn.py`):单个共享权重编码器
   分别应用于产物/donor/acceptor三个图(三者投影到同一个嵌入空间，不像
   TripleGNN每个角色独立权重)，用 h_P - (h_D + h_A) 这个
   Weisfeiler-Lehman差分网络反应向量做组合，而不是拼接(TripleGNN)或图合并
   (CRG)。

两者都复用champion的原子/键特征化(`train_cross_gnn.py`的`af()`/`bf()`/
`graph()`)，确保MAE差异只归因于组合策略而非特征集变化。都在CPU上做了
n=300冒烟测试确认无误后，提交了GPU单seed全量跑(job 26832100 CGR-delta，
26832101 WLDN；gpu_a100，/home/schen3/venv/nequip，跟纯CRG同样的
scaffold-disjoint b973c基线设置，方便跟0.528冠军blend/0.535调优后CRG
做四方对比)。结果待定。

**4-seed mixedtrain公平对比结果**(三个架构都用修正后的34,207/33,407行
训练集，job 26833735-757):

| 架构 | 4-seed均值 | 标准差 | 各seed |
|---|---|---|---|
| 纯CRG | 0.5403 | 0.0086 | 0.5547/0.5327/0.5347/0.5392 |
| CGR-delta | 0.5415 | 0.0141 | 0.5613/0.5251/0.5478/0.5318 |
| WLDN | 0.5413 | 0.0048 | 0.5441/0.5474/0.5350/0.5386 |

修复mixed折入训练集的bug后，纯CRG均值从旧的(少训数据)0.572降到0.540——
之前看起来CRG跟champion差一大截，**大部分其实是训练数据量的混杂因素，
不是架构本身的差距**。修完之后三个架构都落在噪声范围内(~0.540-0.542)，
比champion TripleGNN的4-seed均值(0.531)/blend(0.528)落后约0.01——
跟当初调优后的纯CRG结论("接近但仍略逊"，0.535 vs 0.5295)是同一量级，
现在被另外两个架构独立复现了一遍。

CGR-delta的键级Δ编码改进在这个seed数下**没有**表现出比纯CRG更好
(0.5415 vs 0.5403，完全在噪声内)——这个精细化的额外边特征(未调参下)
没有换来可衡量的收益。WLDN均值跟另外两个打平，但seed间标准差(0.0048)
比CRG(0.0086)和CGR-delta(0.0141)小3-6倍——一个真实的次要发现：
共享权重+差分readout的组合方式看起来对初始化/训练噪声明显更不敏感，
跟MAE本身谁赢无关。

**目前的结论**：CRG/CGR-delta/WLDN三个都还没赢过champion，但都是
站得住脚的架构(尤其WLDN因为稳定性值得记住)，不是死路。如果继续投入，
自然的下一步：(1)给CGR-delta/WLDN也做纯CRG当初做过的那套调参流程
(30-seed默认超参跑一遍→超参搜索→调优后多seed集成，纯CRG靠这个把差距从
0.572收窄到0.535)；(2)把WLDN/CGR-delta跟表格ensemble做blend(纯CRG那边
试过是null结果，但在新的b973c基线下值得重新验证一次)；(3)等Rec-2的
homo+cross统一表落地后重新在联合数据上测三个架构。这次会话不再继续投入
GPU——记录为待续的开放线索，而不是在没有具体新假设的情况下继续烧算力。

**BDE-CRG unmarked对照组跑完了**(job 26829622，8小时重提交后COMPLETED，
用时4:13:05——第一次4小时尝试超时没收敛）。结果：MAE=3.1377，RMSE=5.314，
R²=0.8869，spearman=0.9454——跟marked组几乎一模一样(MAE=3.1404/R²=0.8859，
差0.0027 kcal/mol，noise量级)。

**seed0结论：CRG目标键标记对BDE产物侧预测没有看出可衡量的收益**，跟
cross-benzoin dG那边（标记确实有可衡量效果，虽然不如champion）不一样。
可能的解释(脚本自己文档一开始就标注过这个假说)：BDE每一行本来就只问
"这一行自己的ketC-carbC"/"formyl C-H"(一行一根键，分子类别天然同质)，
标记是冗余信息模型不需要——不像dG-CRG那边一行的图里包含两个反应物各自的
一半，标记才能消歧"新键连的是哪两边"。

已补跑3个seed(marked+unmarked各3个，job 26843134-139，各8小时墙钟)拿
4-seed稳健读数再下最终结论，跟cross-dG那边CRG/CGR-delta/WLDN对比用的
同一套方法论。STATUS.md §9已更新seed0数字和中间推理，最终表格待补。

## 2026-09-20

时隔约3天重新捡起这个项目(最后一条commit是`116ceac`，周四09-17 14:39；
用户确认周四中午后没有主动推进)。实际上有两条线在这段空窗期里已经跑完了
算力，只是没人回来收尾——现在补上，都不需要新跑算力。

**BDE-CRG消融，最终4-seed结论。** Job 26843134-139(seed1-3，marked+
unmarked)早在09-18 01:44-05:45就全部COMPLETED，但落地不到一天就被搁置、
从未分析。合并seed0后：marked均值MAE 3.1779(方差std 0.0640) vs unmarked
均值MAE 3.1374(std 0.0123)——marked反而**更差**0.040 kcal/mol，方向跟"标记
有帮助"相反，且这个差距完全落在marked自身的seed噪声范围内。确认seed0的
结论：**CRG目标键标记对BDE产物侧预测没有可衡量的收益**，B6不采用这个思路。
次要发现：marked组的seed间方差是unmarked组的~5倍——多余的标记信息似乎让
训练对初始化更敏感，跟cross-dG那边WLDN比CRG更稳的发现是同一类现象(方向
相反：那边是"多余结构帮助稳定"，这里是"多余标记增加不稳定")。另外排查
标注同目录下两个陈旧的`*_seed4`文件(日期09-16，早于checkpoint-restore
修复提交`2518fdc`)——`marked_seed4`的MAE=19.6/R²≈0是修复前的已知坏结果，
不属于这次4-seed分析，已明确排除避免以后误读。完整表格见
`pipeline/bde/STATUS.md`§十。这条线到此结束。

**homo SP relabel活动：两条track周四都已跑到100%，但merge一直没重跑。**
直接查shard完成情况：archived track 8972/8972个`.done`标记，regen track
771/771——两条track都在周四(09-17)收尾前跑满了(shards/目录最后写入时间
17:20)，但提高并发之后`merge_homo_sp.py`没有再跑过，磁盘上的
`homo_sp_summary.json`还停留在09-16 09:44的75.4%覆盖率旧快照。现在重跑
merge：覆盖率169,493/184,052(**92.1%**，比旧快照高很多)，QC结论**GREEN**
(`repro_r2scan`对比幸存的3万条旧标签，均值-0.043，标准差2.722，在2.9 kcal
的单构象噪声下限之内，没有系统性偏移)。

直接进入CAMPAIGN_PLAN.md的Phase 2/3——两个提交脚本周四早上就已经预先
写好(`789ffb8`)，不需要改动：提交了`submit_homo_standalone_full.sh`
(job `26947514`，fat_genoa，6小时——组装全库260特征表，用`dG_b973c_kcal`
做Delta基线，然后训练单个XGB champion)和`submit_homo_standalone_gnn_full.sh`
(job `26947515`，fat_rome，24小时，`--dependency=afterok:26947514`——单个
attentive GNN，同样的基线)，串成依赖链让两条腿都能跑完而不需要人盯着。
按CAMPAIGN_PLAN.md的2026-09-10用户指示，两个结果独立汇报，不做blend。
跑完后会把XGB/GNN的MAE补进这条记录。

**用户要求继续自主推进，并问队列是否空闲、26947514/26947515是否需要拆分
加速、记得备份同步。** 查了`sinfo`：`gpu_a100`空闲26个节点，`gpu_h100`空闲
12个。`train_cross_gnn_arch_sweep.py`本身就会自动选`cuda`——GNN那条腿之前
钉在fat_rome纯CPU只是分区选择问题，不是代码限制。取消`26947515`，重新
提交为**`26947605`**，跑在`gpu_a100`（1块GPU，墙钟从24h砍到4h），依赖关系
不变。表格那条腿不需要拆分（XGB本身很快，慢的是132k行上的50折重复CV稳定性
估计，不值得为了缩短它去打断一个正在跑的job）。

**homo全库单XGB结果出来了（job 26947514，merge之后大约1小时COMPLETED）**：
表组装出166,133行×394特征（169,493个可用覆盖的pair里，去掉1,025行没有
split标签的、排除validation split之后剩下的），scaffold-disjoint切分
132,093/17,044/16,996（train/validation/test）。**单XGB Δ模型：holdout MAE
0.684，R² 0.997**（n=16,996；无模型的原始baseline MAE是5.029——Δ模型要
吸收的那个接近常数的~-5 kcal r2SCAN对b973c偏移量）。MLP+XGB ensemble
（只作参考，不是CAMPAIGN_PLAN.md要求汇报的数字）：MAE 0.585。**信这个数字
之前先查了一遍BDE那次的scaffold泄漏陷阱**（`[[bde_scaffold_leakage_finding]]`
memory）——train/test的scaffold集合**完全没有重叠**（train 13,908个，
test 11,934个，0个共有），所以这是一个真正干净的scaffold-disjoint数字，
不是泄漏撑起来的。这个结果接近cross champion的0.528，远好于purge之前的
全库homo模型（MAE≈10）——最可能的原因是homo的pair（donor==acceptor）本身
是个结构上更简单的学习问题（一个分子的身份就决定了整行，不像cross要两个），
加上跟cross一样吃到了b973c基线这个杠杆。值得回头看一下"homo在同等规模下
不比cross难"这条已定结论（PROJECT_PLAN.md §5）——在全量规模+b973c基线下，
homo看起来不只是"不比cross难"，而是**明显更容易**。已备份：模型文件
（`tabular_full/`，6.3MB）直接`git add`了；478MB的全库表被`.gitignore`
（`*.parquet`）挡住而且超过GitHub限制——rsync到了
`~/benzoin_backups/homo_full_library_table_20260920/`，跟`products_all.csv`
一样的处理方式。

GNN那条腿（`26947605`）现在跑在腾出来的GPU上，日志确认了`cuda True`。
跑完后会补进这条记录。
