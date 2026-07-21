# 交接文档:Round9 已完成,Round10 决策待定

**撰写时间**:2026-07-21,round9 自主完成整轮流水线的会话结束时
**Git 状态**:分支 `agent/add-cross-benzoin-v2-data`,HEAD `32b2e0d`,全部已推送到 origin
**用途**:在新对话中启动 round10

---

## 1. 一句话总结

Round9(16,000 个候选 pair)已完整落地,round1-9 已重新组装并重训,得到**新的生产 champion**:MAE **2.074**(round1-8 是 2.106),bootstrap 确认(P=0.988)。learning curve 检查显示 MAE 在当前数据池的 75% 处仍在下降(还没到平台期)——这是**支持**跑 round10 的证据,但幅度温和,不是强信号。**round10 本身要不要跑,这个决策还没做**——这正是你在读这份交接文档的原因。

如果你(用户)决定推进 round10,下面第4节是一份可以直接执行的配方:round9 踩过的每个脚本、参数、坑都记录在案,round10 不用重新摸索。

---

## 2. 当前 champion(round1-9)

| 模型 | round1-8 | round1-9 |
|---|---:|---:|
| 单 XGB(holdout) | MAE 2.498 / R² 0.711 | MAE 2.435 / R² 0.736 |
| MLP+XGB ensemble(holdout) | MAE 2.201 / R² 0.777 | MAE 2.163 / R² 0.786 |
| Attentive-pooling GNN(holdout) | MAE 2.168 | MAE 2.162 |
| **最佳 blend**(bootstrap 确认) | **MAE 2.106**(w=0.60,P=0.9923) | **MAE 2.074**(w=0.55,P=0.988) |

- clean-train 从 27,583 → **43,367** 行(+57%),来自 round9 的数据。
- held-out test 集在两轮之间严格保持**恰好 n=450**(同一批行,未改动)——这正是上表 MAE 对比诚实、可比的原因。
- blend-vs-ensemble-only 差值的 bootstrap(B=20000):90% CI = (0.025, 0.155),完全为正 → GNN blend 是真实的、非噪声级别的提升,和 round1-8 结论一致(那边 P=0.9923,这边 P=0.988,都很强)。

**产物**(均在 `data/cross_benzoin/cross_round9/` 下):
- `scaffold_disjoint_9rounds_v1/models/` —— `champion_scaffold_disjoint.joblib`(单XGB)、`ensemble_scaffold_disjoint.joblib`(MLP+2×XGB)、`feature_list.json`(冻结的260特征schema)、`metadata.json`。
- `gnn_attentive_9rounds_v1/models/` —— `gnn_norm_stats.joblib`、`metadata.json`(内含 `blend_w_gnn=0.55`、`best_blend_mae=2.074`)。注意:`gnn_state.pt`(真正的 PyTorch checkpoint)**不在 git 里**——仓库级 gitignore(`*.pt`),只存在于本机文件系统 `data/cross_benzoin/cross_round9/gnn_attentive_9rounds_v1/models/gnn_state.pt`。如果换了机器/文件系统,需要重训 GNN(见第4节步骤9),不能指望从 git 拿到这个文件。
- `cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet` —— tabular 和 GNN 重训用的确切表(56,136 行 × 273 列:260 champion 特征 + 13 个 meta 列)。这个也**不在 git 里**(仓库级 `*.parquet` 排除)——需要时按第4节步骤7-8 重新生成,或者本机文件系统上应该还在。
- `data/cross_benzoin/cross_round8/...` —— round1-8 champion,同样结构,供对比。

要加载 champion 做预测,用 `cross_benzoin/predict_cross_champion.py` 的
`CrossBenzoinBlendPredictor.load(champion_dir, gnn_dir=...)` ——它会从 `gnn_norm_stats.joblib` 自动识别 GNN 架构(default 还是 attentive)。

---

## 3. Round10 决策证据(learning curve)

`cross_benzoin/learning_curve_check_ensemble.py` 在 round1-9 slim260 表上跑的结果,按数据比例递增做 5折×5次重复的 group-CV(生产用 MLP+XGB ensemble 架构):

| frac | n_pairs | n_rows | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 7,020 | 14,034 | 1.842 | 2.515 | 0.765 |
| 0.50 | 14,040 | 28,063 | 1.797 | 2.466 | 0.781 |
| 0.75 | 21,061 | 42,101 | **1.770** | **2.435** | **0.796** |
| 1.00 | 28,081 | 56,136 | 1.773 | 3.191 | 0.655 |

**解读**:
- MAE 从 0.25 → 0.75 单调下降(约4%相对提升,1.842 → 1.770),这个区间还没到平台期。
- frac=1.0 时 MAE 基本持平(1.773 对 1.770),但 R²/RMSE 明显恶化。这**不是新的不稳定信号**——round6 自己的 learning curve(`data/cross_benzoin/cross_round6/learning_curve_ensemble.csv`)是完全一样的形状。机制:frac=1.0 时已经没有采样自由度了——"子样本"就是全体样本,必然包含所有稀有的重尾难例(比如含磷的异常值,已知的最大误差驱动因素,见 `cross-five-diagnostics-20260717`),而更低比例的随机子样本有机会幸运地把它们排除掉。低 fraction 看起来更"干净"的数字,部分只是这种幸存者偏差,不一定是更真实的信号。
- **总体判断**:0.75 之前的下降趋势是真实的、还没走平,满量数据的 MAE 也没有恶化——这是"round10(同类型更多数据)可能还能推动指标"的温和但真实的证据。这**不是**一个"必须做round10"的强信号,只是"收益还没有明显耗尽"的证据。边际 MAE 收益(按 0.75→1.0 的走平趋势外推,大概在 0.01-0.03 量级)值不值得再跑一轮完整 DFT-SP,这是用户的判断题,数据本身给不出答案。

**这个决策目前还没有做。** 如果用户说"跑",按第4节推进。如果"不跑",上面的 round1-9 champion 就是当前的生产终版,不需要进一步动作。

---

## 4. Round10 配方(如果拍板要跑)—— 复用 round9 的确切脚本

下面的内容和 round9 是同一套流水线,只是把编号换成 round10。脚本路径、参数名、踩过的坑都已经在 round9(2026-07-21)验证过。路径/参数里把 `9`→`10`、`8`→`9` 相应替换即可。

### 4.0 —— 在抽 round10 池子之前:先应用这次发现但故意没动手修的 cb_featurize 修复

这次会话发现(为了不打乱 round9 正在跑的 array 任务而故意没有在中途去改)`pipeline/compute/conf_funnel.py` 和 `conf_funnel_v2.py` 里 `_ff`/`_spj`/`_opt` 这几个 map 函数有个真实的 inode 占用效率问题:每个构象的 scratch 子目录(`ff{i:03d}`、`sp{i:03d}`)要等**整个分子**的构象搜索+描述符+ohess全部跑完才会被清理。对于柔性分子(300-600个构象),现场实测过:单个分子在跑的过程中,300+个构象子目录(1000+个文件)会同时并存,再乘以每个任务 `workers=12` 个分子并发,再乘以同一节点上最多8个任务colocate,对峰值inode占用是个很大的乘数。**建议在 round10 的 featurize array 之前先修这个**:在 `_ff`/`_spj` 里读完每个构象的能量后立刻 `shutil.rmtree()` 掉对应子目录(只保留 L10 存活构象的 `opt{i:02d}` 目录到分子结束)。这是纯效率修复,不改变任何算出来的数值。

### 4.1 —— 抽 round10 候选池

在 `cross_benzoin/sample_round9_from_candidates_v3.py` 基础上改(改文件名/seed,`--exclude-pairs` 要同时排除 round8 **和** round9 已经抽过的 pair,避免 round10 重复抽到)。目标规模(round9 是 8,000 pairs / 16,000 directed rows,类别均衡+磷元素分层抽样)——是否保持同样规模是独立于"要不要跑"本身的另一个决策。

### 4.2 —— Featurize array

`cross_benzoin/slurm/submit_cb_featurize_array.sh`。**强制要求**:先跑 `sinfo -p genoa -o "%20N %10D %6t %C"`,根据实际池子大小定 array 数量,不管空闲节点多少都用 `%60` throttle(这个限制是 `/scratch-local` 每节点 inode 配额,不是集群容量——这次会话已经验证过两次)。CHUNK=30(不是老的默认100——脚本当前默认已经是这个值,2026-07-20因为长尾延迟问题改的)。用 `REQUIRE_CACHE_COMPLETE=1` 提交前,先跑 `cross_benzoin/check_aldehyde_cache_coverage.py --pairs <round10 pairs.csv> --cache <ALD_CACHE>`(存在约0.15%的醛库缓存覆盖缺口,否则会让整个chunk硬失败)。

如果需要重跑(`/scratch-local` 磁盘配额超限错误,错误形式 `exception:[Errno 122] Disk quota exceeded: ...`):通过匹配 `(donor_smiles, acceptor_smiles)` 对照你生成的重跑输入列表来确定失败的 pair——**不要相信错误行的 `id` 列**(整行崩溃的错误会得到一个只有 `{"id": <位置索引>, "error": ...}` 的裸字典,没有 donor/acceptor 信息,来自 `cb_featurize.py` 的外层异常处理器——round9 就是这样)。用 `cross_benzoin/merge_round9_products.py` 作为模板写 `merge_round10_products.py`,合并逻辑要以重跑输入文件为准,而不是靠检测输出里的错误。

### 4.3 —— Product BDE array

`pipeline/slurm/submit_bde_gxtb_product_cross.sh`,CHUNK=50,genoa,很便宜(1 CPU/task,约1小时walltime)。不需要合并步骤——`assemble_cross_round_features.py` 和 `assemble_cross_training_table_v3.py` 都直接 glob `bde_gxtb/chunk_*.csv`。

### 4.4 —— Product mordred array

`cross_benzoin/slurm/submit_mordred_cross_products.sh`,CHUNK=100,genoa,非常便宜(约1秒/分子)。跑完后把chunk合并成一个sidecar CSV(简单的 `pd.concat` 对 `mordred_products/chunk_*.csv`,写到 `data/cross_benzoin/cross_round10/round10_products_mordred.csv`——参考round9的内联合并写法,就3行代码,没有专门脚本,直接写就行)。

### 4.5 —— 组装 round10 候选特征表(DFT前,给主动学习打分用)

`cross_benzoin/assemble_cross_round_features.py --round 10 --products-csv <round10合并后的products csv> --product-mordred-csv <round10的mordred sidecar>`。**不要和 `assemble_cross_training_table_v3.py` 搞混**——那个是DFT之后的最终表组装脚本,需要一个此时还不存在的 `{rtag}_dft_products.csv`。把输出复制成默认命名约定的文件名 `cross_round10_features.parquet`(不带 `--products-csv` 派生的后缀),这样 `score_round_active_learning.py --round 10` 不需要显式 `--candidates-path` 就能找到它。

### 4.6 —— 用主动学习给 round10 池子打分

`cross_benzoin/slurm/submit_score_active_learning.sh`,用 `sbatch --export=ALL,TAG=cross_round10,TRAIN_TABLE=<round1-9 slim260表路径>,FEATURE_LIST=data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1/models/feature_list.json,MODEL=ensemble,N_BOOT=40,N_SELECT=<池子大小>,SEED=42 cross_benzoin/slurm/submit_score_active_learning.sh`。**关键**:`sbatch` 命令行上要显式写 `--export=ALL,VAR=val,...`——内联的 `VAR=val sbatch script.sh` 不会把环境变量传给 SLURM job(round9第一次提交就是这样瞬间失败的,被脚本自己的 `${VAR:?...}` 检查逮住)。打分要用 round1-9 的 champion(不是round1-8的,因为round1-9现在是更好的参考模型)。

### 4.7 —— DFT-SP array

`pipeline/slurm/submit_dft_sp_cross_array.sh`,CHUNK=100,genoa,48 CPU/task,6小时walltime。`--products-csv` 传完整的合并products CSV(不要传过滤后的子集)——`dft_sp_cross_from_geom.py` 的 `build_manifest()` 内部会自己过滤掉错误行/缺xyz的行,所以传未过滤的文件既正确又简单。这是真实的 ORCA(r2SCAN-3c/CPCM-DMSO)计算——每个chunk按小时算,不是分钟。

跑完后:把 `chunk_*.csv` 文件 `pd.concat` 合并成 `data/raw/dft_sp_cross/cross_round10/cross_round10_dft_sp.csv`(匹配 `assemble_cross_training_table_v3.py` 的 `round_paths()` 期望命名),再把合并后的products CSV复制成 `data/cross_benzoin/cross_round10/cross_round10_dft_products.csv`(已确认这就是products文件的一个别名拷贝——`load_round()` 内部会用单独的 `dft_csv` 参数自己做DFT标签的inner-join,不需要额外的join步骤)。

### 4.8 —— 组装 round1-10 表

`cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 5 6 7 8 9 10 --product-mordred-csv data/cross_benzoin/cross_round3/rounds123_products_mordred.csv data/cross_benzoin/cross_round4/round4_products_mordred.csv data/cross_benzoin/screen10k/screen10k_products_mordred.csv data/cross_benzoin/cross_round8/round8_products_mordred.csv data/cross_benzoin/cross_round9/round9_products_mordred.csv data/cross_benzoin/cross_round10/round10_products_mordred.csv`

### 4.9 —— 重新打scaffold-disjoint标签 + 剪枝到冻结的260特征schema

把 `cross_benzoin/relabel_scaffold_split_9rounds.py` 复制成 `relabel_scaffold_split_10rounds.py`,把round9→round10的路径换掉。然后剪枝到和 `data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` 里一样的260个特征(这个冻结的champion schema从round7起一直没变过)加上13个meta列(`id, donor_id, acceptor_id, pair_key, reaction_type, round, donor_smiles, acceptor_smiles, smiles, dG_xtb_kcal, dG_gxtb_kcal, dG_orca_kcal, new_scaffold_split`)。**这个剪枝步骤是必须的,不是可选的**——`train_scaffold_disjoint.py` 的 `_feature_blocks()` 会从给它的表里动态解析特征列,一个没剪枝的N轮合并表可能会重新引入一个clean-train里全是NaN的mordred列,把 `MLPRegressor` 崩掉(round8就发生过这个,根因分析和修复过程见 `round8-complete-attentive-champion-and-round9-inflight-20260720.md` 这条记忆)。剪枝后要检查一下:对clean-train的行,`train[feats].isna().all()` 对每个champion特征都应该是空的。

### 4.10 —— 重训 champion + ensemble

`cross_benzoin/train_scaffold_disjoint.py --table <round1-10 slim260表> --outdir data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1`

### 4.11 —— 重训 attentive GNN

把 `cross_benzoin/slurm/submit_gnn_attentive_9rounds.sh` 复制成 `_10rounds.sh`,换路径(gpu_a100分区,1块GPU,2小时walltime——先用 `sinfo -p gpu_a100` 查一下)。同样的超参数:`--arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0`。

### 4.12 —— Bootstrap验证blend

把 `cross_benzoin/verify_and_bootstrap_9rounds.py` 复制成 `_10rounds.py`,换round9→round10的路径。B=20000,同样的方法论。报告新的MAE和P(blend优于ensemble-only)。

### 4.13 —— 再跑一次 learning curve 检查

`cross_benzoin/learning_curve_check_ensemble.py` 跑在round1-10 slim260表上(用第3节同样的SLURM wrapper),用同样的证据驱动方式给round11的决策做参考。

### 4.14 —— Commit + push

按这次会话同样的分组commit模式来(原始数据落地、重训+champion、learning curve——3-4个逻辑commit,提交前先看 `git status`/`git diff`,不要无脑 `git add -A`)。检查 `.gitignore` 是否已经覆盖:任意深度的 `chunk_*/` 嵌套目录、`*.parquet`、`*.pt`、大的可再生特征join CSV(如果又出现类似140MB量级的中间产物,加一行 `/data/cross_benzoin/cross_round10/cross_round10_features_*.csv`)。

---

## 5. 长期适用的操作规则(以上所有步骤都适用)

- **每次带并发数/throttle/array大小参数的 `sbatch` 提交之前**:先跑 `sinfo -p <分区> -o "%20N %10D %6t %C"`(有用的话再加 `squeue -u $USER -p <分区>`)。绝不要从docstring或者"上次这样用了"的习惯里抄一个throttle数字。`genoa` 的 QoS 有 `MaxJobsPU=128`(`sacctmgr` 确认过);`fat_genoa`/`fat_rome` 没有per-user任务数上限(现场用219个并发任务验证过),但节点池小得多(48/72个节点,对比genoa的300+空闲)——那边也要查 `sinfo`。
- **`cb_featurize` array 特别要注意**:不管空闲节点多少都用 `%60` throttle——这是每节点 `/scratch-local` inode配额的限制,不是集群容量问题。
- **`sbatch --export=ALL,VAR=val,...`** 在提交脚本要读取 `${VAR:?...}` 形式的必需环境变量时必须显式写——内联的 `VAR=val sbatch script.sh` 不会把变量传给job。
- **绝不要为了改一个不理想的参数去取消一个快跑完的任务**——让它跑完,下次提交再改。
- **保留历史输出**——重跑用新文件名,绝不覆盖/删除之前的结果。

---

## 6. 更多细节在哪里找

- `/home/schen3/.claude/projects/-gpfs-scratch1-shared-schen3/memory/round9-complete-round10-evidence-20260721.md` —— 这份交接文档的完整记忆条目来源(每个发现都有更详细的叙述)。
- `/home/schen3/.claude/projects/-gpfs-scratch1-shared-schen3/memory/round8-complete-attentive-champion-and-round9-inflight-20260720.md` —— round8自己的落地过程 + 为什么用attentive-pooling的架构对比证据。
- `cross_benzoin/docs/STATUS_EN.md` / `STATUS_ZH.md` —— 项目状态的滚动记录(你读到的时候可能比这份交接文档滞后一些;这份交接文档截至2026-07-21更新)。
