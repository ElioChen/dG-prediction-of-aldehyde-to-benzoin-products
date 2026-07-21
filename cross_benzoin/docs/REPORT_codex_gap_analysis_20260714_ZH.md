# codex 的 cross-benzoin 工作思考 vs 仓库真实状态 —— 查漏补缺报告(2026-07-14)

对应英文版:`REPORT_codex_gap_analysis_20260714_EN.md`。

## codex 实际做了什么

另一个 Codex CLI 会话(Windows 环境,操作 `ElioChen/dG-prediction-of-
aldehyde-to-benzoin-products` fork)只在 **GitHub / 数据生成层**工作,没有
接触本 HPC 环境的 xTB/g-xTB/ORCA 二进制和 SLURM。它在四轮对话里:

1. 三次生成未标注的定向 cross-benzoin 候选表(v1:12 万条、仅芳香醛 → v2:
   100 万无序对 / 覆盖全部化学类别 → v3:200 万无序对 / 400 万定向行),
   每次都根据用户反馈修正方向(不能预先排除脂肪醛;规模太小;需要覆盖
   全部醛空间)。
2. 给出了一份确实有价值、引用规范的 ML 方法论建议:multi-fidelity Δ-
   learning 的能量分解(`G_high ≈ E_high,SP + (G_GFN2 − E_GFN2,SP)`)、按物种
   相加的 `δE_P − δE_A − δE_B` 校正架构、"homo 是 cross 对角线特例"的框架、
   四种交叉验证场景(homo 对角线 / 训练内新组合 / 单侧外推 / 双侧外推),
   以及建议用 OOF stacking + 不确定性驱动融合取代手工调的 40/60 权重。
3. 提交了 PR #1,内容包括 v3 数据集(Git LFS,约 216 MB)、流式配对分块
   工具(`prepare_pair_chunks.py`)、产物枚举/QC 工具
   (`prepare_product_manifest.py`)、角色感知描述符策略
   (`DESCRIPTOR_POLICY_CROSS.md`)和执行路线图(`NEXT_STEPS.md`)。
4. 正确指出仓库的 `benzoin-dg` CLI 目前仍是 homo-only,在重写 README 时
   没有夸大 cross 推理能力。

方法论本身是合理的,但**大部分对本项目而言并不新**——Δ-learning、不确定
性路由,以及(在上一个会话中)GNN+表格 stacking,在 codex 接触这个仓库
之前就已经在本地建成并验证过了。codex 真正的增量价值在于**候选空间生成
与 cross 专属的规划文档**,这部分是本项目 HPC 侧流水线此前确实没有产出的。

## 本次会话发现并修复的问题

1. **PR 分支如果直接合并,会删除刚确认的 GNN stacking 提升工作。** codex
   的分支(`agent/add-cross-benzoin-v2-data`)是在 `main` 的 `cf870d7` 提交
   ("fix+promote GNN+tabular stacking to full-library scale",job
   24578348)之前切出的。如果照原样合并/rebase,会悄悄删除
   `pipeline/analysis/promote_gnn_stacking_full_library.py`、ChemBERTa/GINE
   纯 SMILES 基线,以及当前真正的生产打分文件
   `products_dG_corrected_GNNSTACK_w40_20260714.csv`(21.8 万行)。已通过把
   `main` merge 回 PR 分支修复(无冲突,两边改动的文件互不重叠),并重新
   推送。PR #1 现在相对 `main` 只是纯增量。
2. **README/状态描述相对已确认结果已过时。** 之前仍把 1.427 的
   GNN+表格融合结果称为"未验证候选",并把 1.503 的表格模型称为"当前
   最佳",但 `cf870d7` 早已在留出测试集上确认了融合结果。已更新,并且
   **明确指出确认后的 1.427 融合结果尚未进入可安装推理路径**:
   `src/benzoin_dG/models/` 目前只有表格模型的 joblib,没有 GNN checkpoint,
   所以 `predict_dG()` 仍然返回 1.503 的模型。这是一个真实存在、尚未解决
   的打包缺口。
3. **g-xTB 失败此前是静默的。** `cb_featurize.py` 对 GFN2 侧失败会写入明确
   的 `error` 字符串(`dG_failed`),但 g-xTB 失败时只是让 `dG_gxtb_kcal`
   留空,没有区分"产物单点失败"和"某个已缓存反应物的 g-xTB 值缺失"。
   已新增 `gxtb_sp_failed` / `gxtb_dG_failed_reactant` 两个错误标签(codex
   在只读审查阶段就指出过这个问题,但它没有二进制环境去实际触发这段代码)。
4. **v3 的 400 万候选是真实的,但仍然零标注**,而且托管在 Git LFS 上,本
   HPC 文件系统未安装 git-lfs。本次没有去搭建 LFS,而是直接从 v3 所依据
   的同一份源库(`data/library/aldehydes_clean_v6.csv`)重新采样——见下节。

## 本次会话真正填上的缺口:真实的 cross-benzoin 产物标签

在本次会话之前,**项目里每一个 cross(donor ≠ acceptor)产物的 ΔG 都是
零**——codex 的 400 万行数据是候选,不是标签;此前本地所有计算
(`homo_v6`、GNN/表格模型、stacking 结果)都只覆盖对角线(`A, A`)。codex
自己的路线图(`NEXT_STEPS.md` Phase 2)其实就是这么要求的:"按 canonical
SMILES 复用已有的醛 GFN2/g-xTB 表;只计算新的产物"。

`cross_benzoin/select_cross_pilot_sample.py` 抽取了一份 **300 个无序对 /
600 条定向反应**的小规模试点样本,在 `{aromatic_carbo, aromatic_hetero,
aliphatic}` 的全部 6 种无序类别组合上均匀分层(donor/acceptor 互换保留为
两条独立的定向记录),并限定为醛侧 GFN2 与 g-xTB 自由能已经缓存在
`data/cross_benzoin/homo_v6/aldehydes_all.csv` 中的分子(220,526/220,859,
覆盖全库 99.9%)。`submit_cb_featurize_array.sh` 新增了
`ALD_CACHE`/`REQUIRE_CACHE_COMPLETE` 参数透传,使这次运行只为真正新增的
部分付费——产物构象搜索、GFN2 优化/频率、g-xTB 单点——醛侧零重算(已验证:
数组任务没有触发"缓存缺失即硬失败"检查)。

**已完成:SLURM job 24607515(array 0–5,genoa 分区,6 个节点)约 48 分钟
跑完,600/600 行,零错误。** 这是本项目历史上第一批真实的 cross-benzoin
ΔG 数值(此前项目里 donor≠acceptor 的 cross 行 ΔG 全部是零,之前所有计算
都只覆盖 homo/对角线情形):

- `dG_xtb_kcal`:n=600,均值 −10.62,标准差 3.69,范围 [−22.16, +7.13] kcal/mol。
- `dG_gxtb_kcal`:n=600,均值 +2.74,标准差 4.37,范围 [−13.43, +29.19] kcal/mol。
- 结构完整性:300/300 个无序对都恰好有 2 条定向记录(无重复、无方向缺失)。
- **全样本 AB/BA 方向性检查**:全部 300 个配对上
  `|dG_xtb(A→B) − dG_xtb(B→A)|` 均值 2.64、中位数 2.16、最大 9.37
  kcal/mol;只有 1.7%(5/300)的配对方向差值接近零(<0.05 kcal/mol)。
  这不再只是零星抽查,而是全样本、定量的确认:donor/acceptor 方向对 ΔG
  是一阶效应,而不是元数据——机理原因(不同方向下哪个碳变成酮碳、哪个
  变成 carbinol 碳不同)见新增方法论文档第 7 节。
- 按类别组合(`dG_xtb_kcal`,各 n=100):aliphatic/aliphatic −11.96
  (标准差 3.29)、aliphatic/aromatic_carbo −10.81(2.54)、aliphatic/
  aromatic_hetero −11.96(4.08)、aromatic_carbo/aromatic_carbo −8.82
  (2.88)、aromatic_carbo/aromatic_hetero −9.79(3.63)、aromatic_hetero/
  aromatic_hetero −10.36(4.33)。含脂肪醛的配对比纯芳香配对平均更放能
  约 2–3 kcal/mol——符合已知的空间位阻/电子效应趋势,不是异常(没有哪个
  类别组合呈现退化/常数分布)。

在监控这次运行的过程中,发现并修复了 `submit_cb_featurize_array.sh` 中
一个真实存在(但本次运行并未触发)的 bug:数组脚本原来的续跑(resume)
检查只判断"`products.csv` 是否非空",而不是"是否每个配对都已经有对应
的行"。`cb_featurize.py` 会随着每个产物完成而增量写入行,因此任何因超时、
被抢占或被重新排队而没跑完一个 chunk 的任务——在更大规模、或分子更慢
(更大/更柔性的底物、开启 Multiwfn、节点资源争用)的情况下是很有可能发生
的——都会留下一个只写了部分内容、但非空的 `products.csv`,原来的检查会
在任意一次重新提交时把它误判为"已全部完成",从而在没有任何报错或警告的
情况下永久丢失未处理完的配对。已修复:续跑检查现在会比较已完成行数与该
chunk 应有的配对数,只有完全匹配才跳过;未完成的 chunk 会被标记并整体
重新计算(`cb_featurize.py` 本身尚不支持行级续跑——续跑会重新付出产物侧
的计算成本,但醛侧仍然走缓存,不会重算)。

可用以下命令重新验证:

```bash
cd /scratch-shared/schen3/benzoin-dg && python3 -c "
import csv, glob
rows = []
for f in sorted(glob.glob('data/cross_benzoin/cross_pilot_v1/chunk_*/products.csv')):
    rows.extend(csv.DictReader(open(f, encoding='utf-8')))
print(len(rows), '行,', sum(1 for r in rows if r['error']), '个错误')
"
```

## 仍然真实缺失的部分(按优先级)

1. **超出这约 600 行试点之外的真实 cross 标签。** 一旦试点验证通过(ΔG 分
   布非平凡、AB≠BA 区域化学正确、失败率低),应按 `NEXT_STEPS.md` Phase 2
   的要求扩展到多样性/不确定性驱动的数千行规模——而不是全部 400 万,这既是
   codex 自己的结论,也是本项目此前的既有结论("不要算全部,用主动学习",
   见 [[delta-mae-noise-floor]])。
2. **把已确认的 1.427 GNN+表格融合结果打包进 `predict_dG()`。** 研究产出
   (全库 CSV)已经存在;可安装的推理路径还没有。
3. **角色感知(donor/acceptor)描述符的实际计算。** *策略*文档已经写好
   (`DESCRIPTOR_POLICY_CROSS.md`);但因为在本次试点之前没有已标注的 cross
   数据可用,cross 专属的描述符表还从未被真正组装过。
4. **cross 的骨架/分子隔离测试集划分**,对应 codex 的 Phase 4 以及本项目
   自己一贯的做法(见 [[data-split-721]])——在(1)产生足够行数之前没有
   实际意义。
5. **PR #1 的描述文字已过时**(仍停留在合并前、试点前的状态)。本次会话
   无法更新它:本 HPC 环境未对 `gh` 完成认证(只有基于 SSH 的 `git push`
   可用,身份为 `ElioChen`)。分支和提交本身是最新的,只有 PR *描述文本*
   需要手动刷新,或者在交互式会话中执行一次 `gh auth login`。
