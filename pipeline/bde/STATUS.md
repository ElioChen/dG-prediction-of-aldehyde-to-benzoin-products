# BDE 预测子课题 — 状态总览(STATUS)

> **这是本子课题的权威入口文档,新会话/协作者请先读本文件。**
> 长期维护:每次有实质进展就更新本文件的「五、当前状态」和「六、下一步」两节。
>
> - 完整方案设计(Phase 1–4、模型矩阵、标签体系):仓库根目录 `BDE_prediction.md`
> - 逐日工作日志(踩坑记录、每步怎么做的):`pipeline/bde/PROGRESS_20260714.md`
> - 可读结果快照(2026-07-16,部分数字已被下文修正):`pipeline/bde/RESULTS_SUMMARY_20260716.md`
> - 本文件创建于 2026-09-02(sonnet-5 会话),用于在 2026-07 scratch purge 之后
>   把分散在上述三份文档里的结论重新收敛成一份可持续维护的状态页。

---

## 一、这个子课题在做什么

**核心问题**:能否用 ML 直接预测 BDE 本身,替代昂贵的 g-xTB / DFT 计算?

两个预测目标(标签均为本项目自己的 g-xTB 计算,r2SCAN-3c/CPCM(DMSO) 几何一致):

| 代号 | 目标键 | 化学意义 |
|---|---|---|
| Task A(醛侧) | 醛 formyl **C–H** BDE | benzoin 缩合的第一步断键 |
| Task B(产物侧) | 产物中心 **ketC–carbC** BDE | benzoin 产物的关键 C–C 键 |

**与隔壁会话的边界(重要,勿越线)**:本仓库同时被两个并行会话使用——

- **本会话 = BDE 预测**(直接把 BDE 当预测目标)。
- **隔壁会话 = cross-benzoin ΔG 预测**(Δ-learning,`dG_pred = dG_gxtb + ML 修正`,
  把已算好的 g-xTB/DFT 产物 BDE 当作**输入特征**,不预测 BDE 本身)。

不要在本会话推进 `cross_round{N}` 的采样 / 特征化 / DFT-SP / 组表 / `train_cross_delta.py`
重训——那是隔壁会话的事。本会话对 cross 数据的合法用途只有一个:**消费隔壁每一轮产出的
cross 数据,做 BDE 侧的 homo→cross 迁移实验**。详见 `memory/concurrent_cross_bde_session.md`。

---

## 二、模型排名与诚实数字

Phase 1 的 7 个基线(B0–B6)已全部跑完(2026-07-14 ~ 07-20)。

### 2.1 冠军:B6 = D-MPNN + 局部 3D 描述符融合

`pipeline/bde/train_gnn_hybrid_bde.py`。同一个 `BondMessagePassing` 图编码器(和 B4 一样),
把 H-SPOC 那套局部电子结构描述符(Fukui、WBO、ADCH/QTAIM 电荷、vbur、sterimol……)通过
chemprop 的 `x_d` 通道拼接到图嵌入之后、FFN 之前——直接照搬主 dG 生产模型 `gnn_core.py`
已验证过收益的融合模式,零新增计算。

**诚实的 champion 数字用 scaffold-disjoint 划分(不是 molecule-cold-split):**

| 划分方式 | 醛侧 MAE / R² | 产物侧 MAE / R² | 说明 |
|---|---:|---:|---|
| molecule-cold-split(**已废弃**) | 1.104 / 0.900 | 2.076 / 0.921 | 87.6% / 83.9% 的测试集 scaffold 也出现在训练集,数字虚高 |
| 事后「新 scaffold」估计 | 1.691 | 3.451 | 系统性高估真实退化 |
| **真·scaffold-disjoint 重训** | **1.579 / 0.843** | **3.060 / 0.886** | **对外一律引用这组数字** |
| 相对退化 | +43% | +47% | |

- 从零重训得到的 1.579 与事后估计 1.691 相互印证(~7% 差),确认「醛侧 BDE 预测有真实、
  实质的 scaffold 泛化差距」不是比较 bug 的假象。job 24694779。
- 带 checkpoint 的复训(job 24706282):醛 1.591 / 0.838,产物 3.032 / 0.888——种子间正常波动。
- 有意思的反差:隔壁 cross-benzoin 自己的 scaffold-disjoint 重训 R² 反而**改善**
  (0.730→0.763)。**同一份醛库、相似方法学,两个模型/任务结论相反 → scaffold 泄漏的影响
  必须逐模型/逐任务实测,不能假设可迁移**。

### 2.1b 2026-09-06 全量(220k)重训 —— 取代 2.1/2.2 的旧数字

`bde_post_sweep`(job `26326313`→`26418249-53`)在描述符库从 42k(purge 后局部恢复)
重建到**完整 220k**(`assemble_homo_descriptor_libs.py`:products_all 184,199 行,
aldehydes_all 209,526 行,见 §五)之后,在全量数据上重跑了 B6 ckpt、B6 5-seed deep
ensemble、B4/B5 honest scaffold-disjoint、GBM bakeoff。**以下数字取代 2.1/2.2 里
基于 42k 局部库的旧数字(1.579/3.060 那组)——数据集规模变了,不是同一个实验,不要
再引用旧数字。**

| 模型 | 醛 MAE/R² | 产物 MAE/R² | 备注 |
|---|---|---|---|
| **B6 5-seed deep ensemble(新冠军)** | **1.851 / 0.797** | **2.826 / 0.899** | 比单 seed 低 ~12%;`sigma~|err|` spearman 0.42/0.40,可做 route-to-DFT:keep80% kept_MAE 1.27/2.18 |
| B6 单 checkpoint | 2.094 / 0.768 | 3.192 / 0.883 | 5 个 seed 全部落在 2.07–2.13 / 3.18–3.24 窄带,单 seed 已接近稳定 |
| B4 D-MPNN(纯 2D 图,honest split) | 2.213 / 0.757 | 5.263 / 0.758 | 产物侧比 B6 差 65%——`x_d` 融合优势在全量诚实划分下依然稳健,甚至比旧的小数据集结论更悬殊 |
| B5 BonDNet 式 | 2.632 / 0.722 | 5.082 / 0.787 | 同上 |
| GBM 最优头(RF 醛 / HistGBM 产物,H-SPOC 特征全量) | 3.494 | 5.317 | 远逊于 B6,GNN 端到端优势在全量下更明确 |

未跑:Phase-3 3D 反应差分模型(SchNet/DimeNet,`submit_gnn3d.sh`)——sweep 脚本没有
自动提交这一项(计划里的第 5 项,`POST_ARRAY_MODEL_SWEEP.md` §1 有记录),需要人工决定
是否补跑(几何已保留在 `chunk_*/geom.tar.zst`,不需要重算)。鉴于 B6 已经把 B4/B5/GBM
甩开这么多,3D 模型能再突破的先验概率不高,不建议无人值守自动跑。

### 2.2 旧:小数据集(42k 局部库)排名(仅存档,不再引用)

| 模型 | 醛 MAE/R² | 产物 MAE/R² | 一句话 |
|---|---|---|---|
| B6 GNN+3D 描述符融合(旧冠军) | 1.579 / 0.843 | 3.060 / 0.886 | 比 B4/B5 低 32–42% MAE,`x_d` 融合优势稳健 |
| B4 D-MPNN(纯 2D 图) | 1.604 / 0.830 | 3.641 / 0.834 | 图结构很强,但看不到局部电子结构(naive-split 数字,scaffold-disjoint 复训见 PROGRESS〇-11) |
| B5 BonDNet 式(反应差分图嵌入) | 1.923 / 0.802 | 3.689 / 0.846 | 与 B4 互有胜负 |
| H-SPOC(局部 3D 描述符 + XGB,调参后) | 2.42 / 0.758 | 4.05 / 0.818 | **零新增计算**,性价比最高的基线 |
| D-SPOC-217(RDKit 全局描述符差) | ~ / 0.629 | 5.365 / 0.715 | |
| B2 ECFP + XGB | 3.49 / 0.563 | 6.18 / 0.652 | 传统 2D 指纹基线 |
| B0 ALFABET zero-shot / B1 微调 | 近乎无区分力 | 输出方差趋零 | 域外物质,退化成预测训练集均值 |

**架构层面唯一稳健确认的结论**:B6 相对 B4/B5 的 `x_d` 融合优势(35–47 个百分点)。
细粒度架构改动(attentive pooling、chemprop 2.3 的 MAB)在 naive-split 上看到的小幅优势,
在 scaffold-disjoint 下**不再成立**——它们也是泄漏驱动的假象。

---

## 三、关键方法学结论

1. **`x_d` 融合有效**(见 2.1)。这不是本项目原创,是复用主 dG 模型已验证的模式。

2. **homo→cross 迁移:数据够多时「仅 cross 训练」会追平「homo 预训练+微调」。**
   round1-7 规模(cross 训练 18,025 行,scaffold-disjoint):

   | 方案 | MAE | R² | ρ |
   |---|---:|---:|---:|
   | zero-shot(仅 homo) | 6.362 | 0.537 | 0.755 |
   | C:homo 预训练 + cross 微调 | 4.501 | 0.695 | 0.852 |
   | A:仅 cross 训练 | 4.524 | 0.703 | 0.858 |

   round3(cross 2322 行)时 C 明显领先 A;round5(10107 行)时差距收窄到噪声量级;
   round1-7(18025 行)时**基本消失**。**实用结论:round8 及以后只求最佳 BDE 精度,直接
   cross-only 训练即可**,不必再跑两阶段流程——除非未来出现结构性稀缺的官能团子集。
   naive 合并(homo+cross 直接拼)在所有规模下都是最差的,不要用。

3. **g-xTB 标签的系统性误差主要来自 single-point 方法层级,不是几何优化。**
   DFT 仲裁 round2(n=111):Δ_SP 均值 −21.0(std 12.7)vs Δ_geom 均值 1.58(std 3.3)。
   → 要真正缩小 g-xTB 相对 DFT 的 ~14–16 kcal 差距,得升级 single-point 方法本身,
   动几何设置没用。round1 报告过的「磷元素几何异常」在 round2(磷组 n=13)**未复现**,
   属 small-n artifact,已弃用。

4. **g-xTB 标签的构象噪声下限 ~2.1–2.7 kcal/mol**(单构象)。B6 醛侧 MAE 1.58 距此下限
   还有空间;产物侧 3.06 也未触底,但差距更小。

5. **nitro 官能团的 Δ_SP 呈双峰分布(未解释,开放项)**:19 个 nitro 分子里 13 个 Δ_SP
   强负(~−22),5–6 个为正(~+19)。RDKit 层面的候选解释变量(是否含硫、环数、重原子数)
   都对不上。不建议在证据不足时硬编机理解释——标记为待查。

6. **残差集中在重杂原子 / 强 EWG 官能团**(硒、硝基、N-氧化物、磷、亚胺;产物侧还有三氟
   甲磺酸酯、磺酰基、酰胺)——与主 dG 模型发现的最难官能团一致,提示部分残差是 g-xTB 标签
   噪声而非模型容量问题。

---

## 四、交付物 / 封装状态

- `pipeline/bde/predict_bde_champion.py` — `BDEChampionPredictor` 类,单文件推理封装,
  镜像 `cross_benzoin/predict_cross_champion.py` 的设计。曾用 SLURM(job 24764832)在
  完整 held-out 测试集上精确复现训练数字(醛 MAE 1.5913、产物 MAE 3.0319)。
  封装时修过一个真实 bug:复现测试集时漏了 `qc_filter` 这一步。
- **⚠️ 该封装依赖的 B6 checkpoint(`runs/logs/scaffold_disjoint_bde/models/b6_*.pt`)
  已在 2026-07 purge 中丢失**(gitignore 的 `.pt`,从未入库)。当前 `predict_bde_champion.py`
  有代码没权重。见下节。

---

## 五、当前状态(2026-07 scratch purge 之后)

2026-07-20 前后 Snellius scratch 被清空。live 仓库现为
`/gpfs/scratch1/shared/schen3/benzoin-dg-restored`(branch `agent/recovery-20260902`,
recovery commit `d1804ea`)。旧的 `/gpfs/scratch1/shared/schen3/benzoin-dg` 是空壳。
详见 `RECOVERY_REPORT_20260902.md` 和 `memory/benzoin_dg_repo_moved_and_data_loss.md`。

**recovery commit 主要服务隔壁 cross-benzoin dG 项目,BDE 侧只是搭便车恢复了一部分。**

### BDE 侧数据资产盘点

| 文件 | 状态 | 来源 |
|---|---|---|
| `aldehydes_bdfe_gxtb_descriptors.csv`(醛 BDE 标签,220,522 行) | ✅ **bit-exact 恢复** | home 备份 `homo_v6_scratch_archive/bde/bdfe_gxtb` 逐块重拼 |
| `aldehydes_mordred_slim102.csv` | ✅ **bit-exact 恢复** | 同上 |
| `aldehydes_all.csv`(醛局部 3D 描述符) | ⚠️ **仅部分重算 42,336 行**(原 ~220k) | recovery 只补了 round10 pairs + rounds 1-7 醛;`cb_featurize.py --aldehydes-only`;保真度中位相对偏差 ≤0.03% |
| `products_bdfe_gxtb_descriptors.csv`(产物 BDE 标签) | ❌ 缺,**可从 home 备份重拼** | `homo_v6_scratch_archive/bdfe_gxtb_products.tar.gz` 有 `id,bde_gxtb_kcal`(2206 块) |
| `products_all.csv`(产物局部 3D 描述符) | ❌ 缺,**无备份,需全量重算** | 无任何归档;成本约为醛侧的 ~6×;和 `aldehydes_all.csv` 的处境一样 |
| `aldehydes_scaffold_split_from_dG.csv` / `products_scaffold_split.csv` | ✅ 入库,完整 | git tracked |
| B6 checkpoints `b6_{aldehydes,products}_scaffold_disjoint.pt` | ❌ **永久丢失** | gitignore 的 `.pt`,从未入库;只能重训 |
| `runs/logs/scaffold_disjoint_bde/*.json` 结果 + 预测 CSV | ❌ 丢失 | 结论已在本文件/PROGRESS 中留存,原始文件没了 |

### ✅ 2026-09-06 更新:已一键复现 + 超越(见 §2.1b)

上面这节(§五 BDE 侧数据资产盘点)描述的是 2026-09-02 purge 后的残局;截至
**2026-09-06**,这个缺口已经补齐:`bde_homoprod`(genoa,`26316404`,2209 任务)+
`bde_homoprod_rome`(`26324801`,1259 任务)两个 featurize array 全部跑完
(COMPLETED 2191+1191,FAILED 18+68——失败率均 <6%,可接受),`bde_post_sweep`
(`26326313`)自动装配出**完整**的 `aldehydes_all.csv`(209,526 行)和
`products_all.csv`(184,199 行,首次拥有全量),并在此基础上重训了 B6
ckpt+deep-ensemble、B4/B5、GBM——数字见 §2.1b。B6 checkpoint 也已重新产出并
`git add -f`(`runs/logs/scaffold_disjoint_bde/ensemble/models/*.pt`,不再"永久丢失")。
路径硬编码问题在这轮 sweep 里已经绕过(sweep 脚本用的是 restored 仓库路径)。

原始的"无法一键复现"诊断(醛侧描述符卡在 42k、产物侧全缺、路径硬编码)保留在下面
作为历史记录,不再是当前状态:

- **醛侧**(历史):标签有(220k),但局部描述符 `aldehydes_all.csv` 只有 42k 行——
  `train_gnn_hybrid_bde.py` 做 `labels.merge(mol, on="id", how="inner")`,有效训练集被
  卡在 ~42k,远小于产出 champion 数字时的 n_train=188,254。
- **产物侧**(历史):标签和局部描述符都缺,标签可重拼、描述符要重算。
- **另有路径问题**(历史):`train_gnn_hybrid_bde.py` 里 `H` 硬编码指向已被清空的
  `/scratch-shared/schen3/benzoin-dg/.../homo_v6`,重训前需改指到 restored 仓库路径。
- 多个 `pipeline/slurm/submit_*.sh` 的 `PY=` 仍指向已清空的 `envs/bde_gnn`,工作区里有
  未提交的修复 diff(把 `PY` 改成 `${PY:-/home/schen3/venv/nhc-workflow/bin/python}`)。
  **注意该 diff 与隔壁会话共享同一个 checkout,未确认前不要替隔壁提交。**

---

## 六、进行中的恢复工作(2026-09-02,用户指示 A+B+C 全做,CPU 用 genoa)

### ✅ 已完成(无需大算力)
1. **产物 BDE 标签重拼**:`pipeline/bde/rebuild_products_bde_labels.py` 从 home 备份
   `bdfe_gxtb_products.tar.gz`(1460 块)重拼出
   `products_bdfe_gxtb_descriptors.csv`(218,966 行,`bdfe_gxtb_kcal,id,bde_gxtb_kcal`)。
   raw 有极端离群点,`train_gnn_hybrid_bde.py` 的 `qc_filter` 会过滤。已 `git add -f`
   入库(commit `ca15188`)。
2. **路径修复**:`train_gnn_hybrid_bde.py` 的 `H` 改为 repo 相对解析
   (`parents[2]/data/cross_benzoin/homo_v6`)+ `$BDE_HOMO_V6` 覆盖 + 老绝对路径兜底。
   其余 bde 脚本仍硬编码老 `/scratch-shared/...` 路径(见文件头 grep),用到时再逐个改。
3. **chemprop/torch 环境重建**:旧 `envs/bde_gnn` 是空壳。新建
   `/home/schen3/venv/bde_gnn`(py3.11 + torch 2.13.0+cu130 + chemprop 2.2.0 +
   lightning 2.6.5,与旧环境 torch 版本一致),MPNN 构建已验证通过。
   `submit_b6_scaffold_disjoint_ckpt.sh` / `submit_bde_scaffold_disjoint.sh` 的 `PY=`
   默认已改指到它。
4. **git**:3 个 commit(`6824063` env repair 采纳、`ca15188` BDE 重建基建、`94873ba`
   cross rounds-1-7 恢复脚本),分支 `agent/recovery-20260902`,**尚未 push**。

### 🟡 进行中 — 选项 B 描述符库重算(genoa)
- **SLURM array `26316404`**(job name `bde_homoprod`,genoa,2209 tasks,CHUNK=100,%96):
  `pipeline/slurm/submit_bde_homo_product_featurize.sh` 跑 `featurize_product.py` over
  全部 220,859 个 homo pair(donor==acceptor),`--multiwfn --emit-aldehydes`。
  **一趟产物 campaign 同时产出两个库**:产物局部描述符(`features.csv`)+ 方法一致的
  醛局部描述符(`aldehydes.csv`),省掉单独的醛 backfill。
  输入/输出:`data/cross_benzoin/bde_homo_product_featurize_20260902/`。
  smoke(job 26316093,8 pairs)已验证全工具链跑通、B6 所需 43 个产物特征齐全非 NaN。
  预计墙钟 ~10–15h(视 genoa 竞争)。
- **完成后**:
  1. `python pipeline/bde/assemble_homo_descriptor_libs.py --chunk-dir
     data/cross_benzoin/bde_homo_product_featurize_20260902` → 写
     `products_all.csv`(product_smiles→`smiles`,donor_smiles→library id)+
     `aldehydes_all.csv`(canonical-SMILES→library id remap,partial 42k 兜底)。
     注意:`features.csv` 没有 `dG_gxtb_kcal`(只有 `dG_xtb_kcal` ALPB),B6 BDE 训练
     用不到 dG,但 `headroom_probe.py` 等分析脚本会缺这列。
  2. scaffold-disjoint 重训 B6 醛+产物,带 `--save-checkpoint`
     (`pipeline/slurm/submit_b6_scaffold_disjoint_ckpt.sh`,先修其 `PY=`)。
  3. 结果 JSON/pred CSV/`.pt` 全部 `git add -f`,并加进
     `submit_backup_recovery_artifacts.sh` 的归档清单。

  **→ 完整的阵列后 model sweep 计划见 `pipeline/bde/POST_ARRAY_MODEL_SWEEP.md`**
  (2026-09-02 晚扩写:除 B6 ckpt 外还有 B6 5-seed deep ensemble + B4/B5 诚实重训 +
  GBM head bake-off + Phase-3 3D 模型;Monitor task `bm8632d2g` 盯阵列出队)。

### 🟢 2026-09-02 晚 — sweep 前置已就位(commits `826c0e9`, `ea7936e`)
- **float-id 修复(`826c0e9`)**:purge 恢复把 `aldehydes_bdfe_gxtb_descriptors.csv` /
  `aldehydes_scaffold_split_from_dG.csv` / `aldehydes_bde_alfabet.csv` 的 id 写成了
  `2.0` 浮点格式,而重建的 `*_all.csv` 用 `2`。字符串 merge 会 **静默 inner-join 到 ~0 行**
  —— 阵列后所有 B4/B5/B6 重训都会当场废掉。已加 `qc.norm_id()` 并在 4 个 trainer 的每个
  id 列 read 后套用(**不改共用 CSV**,隔壁窗口也在读)。验证:label↔descriptor 重叠
  0 → 42,307。隔壁窗口若也遇到 `2.0` id join 问题,同样用 `norm_id` 或先规范化再 join。
- **增量几何归档(staging job `26324608`,`archive_completed_chunk_geoms.sh`)**:每个
  完成的 chunk 的 `xyz/`+`ald_xyz/` 打包成校验过的 `geom.tar.zst` 再删散文件,跑的过程中
  持续回收 inode(~200/chunk)。**取代 HANDOFF §2.2.2 的 `rm -rf`**——用户要求压缩保留
  几何(Phase-3 3D 模型要用)。~3.3× 体积压缩,完成时 ~45 万 inode → ~2200。
- **GBM head bake-off 首screen**(partial 42k,7 个 adch_/qtaim_ 特征在 partial 库里还是
  常数,所以绝对 MAE ~3.5 偏高):HistGB 3.501 / ExtraTrees 3.542 / RF 3.599 / XGB 3.630。
  → 全量重跑时把 **HistGradientBoosting** 一起带上(比现任 XGB 稳定低 ~3–4% MAE,还快)。

### 🟡 进行中 — 选项 C 分析
- **nitro 双峰 Δ_SP**:已细查(见 `pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md`)。
  结论:18 个 nitro 里 5 个 Δ_SP 为正的,`bde_gxtb_kcal` 均值 101.6(vs 负组 69.0),
  其中 2 个 g-xTB BDE >100 kcal/mol(醛 formyl C–H 物理上不该这么高)——更像是
  **g-xTB 在这几个 polynitro/脂肪链 nitro 上自身电子结构失败产生的标签离群点**,
  不是真正的 single-point level-of-theory 双峰。判别量是 `bde_gxtb_kcal` 本身,不是全局
  结构描述符(与 PROGRESS 里"结构描述符解释不了"一致,补上了机理)。
- **homo→cross 学习曲线补点**:待隔壁 round8/9/10 的 cross BDE 数据可用后再做
  (round8/9 的 DFT 标签在 purge 中丢失、无副本,见
  `memory/benzoin_dg_repo_moved_and_data_loss.md`;round10 还在隔壁 featurize)。暂挂起。
- **Phase 3 3D 反应差分模型**:`pipeline/analysis/gnn3d_schnet_dimenet.py` 已存在骨架,
  待描述符库重建完再评估。**几何现在会保留**(见上,`geom.tar.zst`),不再需要为此
  单独重算几何 —— sweep step 5,先把 `chunk_*/geom.tar.zst` 解到平铺目录。

---

## 七、防止再次丢失

`memory/scratch_disk_quota_risk.md` + `RECOVERY_REPORT_20260902.md` §7 的教训很具体:
**凡是同时被 gitignore 且没有 home 备份的文件,一次文件系统策略变更就没了。**
BDE 侧要补的备份钩子:`aldehydes_all.csv`、`products_all.csv`、`b6_*.pt`、
`runs/logs/scaffold_disjoint_bde/` 全部结果——重建后立即 `git add -f` 或纳入
`submit_backup_recovery_artifacts.sh` 的归档清单。
