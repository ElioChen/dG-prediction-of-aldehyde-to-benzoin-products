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

--- 快照（2026-09-15）：homo SP 是唯一在跑的计算，ETA ~09-17/18，除了等待和监控没有
别的事。cross-benzoin 这条线目前没有已知缺口——两个冠军基线（g-xTB、b973c）都已部署、
已校准，文档与代码实际运行的东西一致。下一个设计步骤不变：flying dataset 构建顺序
第 4 步（把标签/切分/基线接入读取 API），或者等 homo SP 出来为 Rec-2 统一提供输入。 ---
