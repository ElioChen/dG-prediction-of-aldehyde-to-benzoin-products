# 化学空间 & Flying Dataset —— 规范

> 状态：**规范 2026-09-10 写完；构建顺序第 1-3 步 2026-09-11 完成**（`aldehyde_index.parquet`
> 已冻结 + `chemical_space.py` 的 `pair(i,j)` 特征路径已写并校验，见 §8）。2026-09-10
> 按用户要求写（"cross 的建库之前根本不对，真正的
> 化学空间应该是 220k 的平方 … 需要知道 flying dataset，方便以后读取以及模拟、预测"）。
> 见 `PROJECT_PLAN_ZH.md` §2.11。English: `CHEMICAL_SPACE.md`（保持同步）。
>
> 目的：一次性、无歧义地定义 **benzoin cross ΔG 化学空间是什么**，以及**以后每一步
> 怎么读它** —— 标注、模拟、预测、主动学习采集、Goal-3 筛选 —— 而**永不落地一张全部
> 对的表**。

---

## 1. 旧候选池为什么错

cross 模型的训练/筛选是针对 `candidates_v3`（~124 万有向对）。那是一个**构造出来的
子集** —— 来自一个有界生成器（MaxMin 多样性挑选、类别分层、固定 reservoir），从来
不是真实空间。所有"筛选全库"的说法因此都只限于一个任意样本，逐轮 AL 采集也是在这个
任意样本里挑。

**真实空间**是：任取一个醛做 donor，任取一个醛做 acceptor。donor 和 acceptor 化学
角色不同（donor 变成酰基负离子 / Breslow 碳；acceptor 是亲电体），所以**有向**对
是诚实的计数。

---

## 2. 空间，精确定义

| | |
|---|---|
| **基础库** | `data/library/aldehydes_clean_v6.csv` —— **220,859** 行（CSV 220,860 行含表头，2026-09-11 修正 off-by-one）（单醛，过滤后，带 `cho_class` + `xtb_risk`）。设计上包含算不出来的分子（见 `PROJECT_PLAN_ZH.md` §1.2）。 |
| **homo 空间** | 220,859 个对角对 `(i, i)`。 |
| **cross 空间** | 所有有向 `(i, j)`：**220,859² = 4.878 × 10¹⁰** 有向对（488 亿）。无向 ≈ 2.44 × 10¹⁰。 |
| **已落地** | ~35,528 对有 DFT 标签（已暂停的 AL 1–10 轮）。其余全是虚拟的。 |

任何文件都不会存全部对。一个对是一个**地址**，按需计算。

---

## 3. 醛的规范索引（必须先冻结）

- **`ald_idx`** ∈ `[0, 220859]` —— 稳定整数，一次性分配，取自 `aldehydes_clean_v6.csv`
  一份**冻结**副本的行序。永不重派，永不是对过滤视图的 `enumerate()`。
- 一起带：canonical SMILES（RDKit canonical，单一固定协议）、`InChIKey`（人类可读次级
  键）、`cho_class`、`xtb_risk`、Bemis–Murcko scaffold SMILES、`computable` 标志
  （一旦确认某分子几何/DFT 失败就置 false，下游直接跳过不再重试）。
- 产物：`data/chemical_space/aldehyde_index.parquet` —— **2026-09-11 已冻结**
  （`cross_benzoin/build_aldehyde_index.py`）。这个文件是唯一真值源；`homo_v6/*`
  缓存要重新 key 到它（构建顺序第 2 步，见 §8）。

---

## 4. 每醛缓存（算一次，给 ~10⁶ 个对复用）

按 `ald_idx` key。多数已在 `data/cross_benzoin/homo_v6/` 下，只是索引不一致：

| 缓存 | 内容 | 来源 | 状态 |
|---|---|---|---|
| 几何 | GFN2 `--ohess` xyz + `xtbopt` | funnel_v3 | ✅ ~18 万（归档不全） |
| xТБ 能量 | `E_el`、`G`（Gibbs）、thermal `G−E_el` | xtb | ⚠️ 有，但 `G_xtb` 对 ~8% 行错位 —— **按 §1.2 规则界检** |
| 局部 QM 描述符 | 前线轨道、Fukui、Mulliken/ADCH、WBO、QTAIM、Vbur、Sterimol（260 schema 里 ~35% 的"局部 QM"） | `ald_descriptors_qm.py` | ✅ `aldehydes_all.csv` 209,526 |
| Mordred（slim） | 2D/3D 描述符，精选子集 | mordred | ✅ `aldehydes_mordred_slim102.csv` |
| BDE | formyl C–H BDE（ALFABET + g-xTB） | `pipeline/bde` | ✅ ~22 万 |
| g-xTB / B97-3c 单点 | Δ-learning 基线用 | 本项目 | 🔄（逐对，见 §6） |

**原理：** 醛是昂贵计算的单元。一个对的特征 = （donor 缓存）⊕（acceptor 缓存）
⊕（少量产物侧 + 交互项）。所以整个 4.9 × 10¹⁰ 空间的成本 ≈ ~22 万次醛特征化 +
便宜的逐对拼装，而不是 4.9 × 10¹⁰ 次任何东西。

---

## 5. 读取 API（"flying dataset"）

一个薄模块 —— `chemical_space.py` —— 暴露一个惰性视图：

```
space = FlyingDataset(index="data/chemical_space/aldehyde_index.parquet",
                      caches="data/cross_benzoin/homo_v6/",
                      labels="data/chemical_space/dft_labels.parquet")

space.n_aldehydes                      # 220859
space.pair(i, j)                       # -> dict:
    {  donor_idx, acceptor_idx,
       donor_smiles, acceptor_smiles, product_smiles,   # 产物由反应模板生成
       reaction_type,                                   # 由两个 cho_class 得
       features: np.ndarray[260],                       # 从缓存惰性拼装
       baseline_gxtb, baseline_b973c,                   # 未算则 None
       scaffold_split,                                  # 骨架不相交，由两个 scaffold 现算
       label_dG,                                        # 不在 dft_labels 里则 None
       computable }                                     # False -> 调用方跳过

space.iter_pairs(donor=None, acceptor=None, where=...)  # 生成器，永不成 list
space.sample(n, strategy=..., seed=...)                 # 给 AL / 筛选出 batch
space.features_batch(pairs)                             # 一批对的向量化拼装
```

- **产物 SMILES**：反应模板（`featurize_product.build_product` / `FP.build_product`，
  流水线用的同一个）—— 由 (donor, acceptor) 确定。
- **特征拼装**：复用 `assemble_cross_training_table*` 的列逻辑，重构成对一个对、从缓存
  dict 操作，而非对一张合并表。
- **scaffold_split**：一个对是 `train` 当且仅当两个醛的 scaffold 都**不在**留出 scaffold
  集；两个都在 → `test`/`validation`；否则 `mixed`（排除）。留出 scaffold 集冻结在索引文件里。
- **磁盘上没有对表。** `iter_pairs` / `sample` 产出地址；拼装按需、最多短暂缓存。

---

## 6. DFT 标签存储

- `data/chemical_space/dft_labels.parquet` —— 每个已算的对一行：
  `donor_idx, acceptor_idx, dG_r2scan_kcal, dG_b973c_kcal, dG_gxtb_kcal,
   geom_hash, method_tag, campaign, date`。
- 按 `(donor_idx, acceptor_idx)` 地址 key —— **不是** InChIKey 对字符串，**不是**
  `candidates_v3` 行 id。
- **迁移：** 把现有 35,528 个有标签对（1–10 轮 + Tier B）通过 canonical SMILES →
  `ald_idx` 查表映射到地址；保留（是有效数据），tag `campaign="al_r1_10"`。

---

## 7. 用起来是什么样

- **重做 cross AL**（`PROJECT_PLAN_ZH.md` §2.7）：采集函数对
  `space.sample(n_candidates, strategy="stratified")` 打分、排序、把下一批 DFT 选成
  地址，`run` 标注它们，`dft_labels.parquet` 增长。不用 `candidates_v3`。
- **Goal-3 筛选**：`space.iter_pairs(where="dg_favorable & ~baseline_risk")` 流式过
  `predict_dg`，按 conformal 上界 top-k → 候选清单。
- **模拟**（未来任何 DFT 战役）：吃一份地址清单，从缓存取几何，算，写回 `dft_labels`。

---

## 8. 构建顺序（每步小、理解清楚、验证过）

1. ✅ **2026-09-11 完成。** 冻结了 `aldehyde_index.parquet`（220,859 行；
   `ald_idx`、canonical SMILES、`InChIKey`、`cho_class`、`xtb_risk`、scaffold；
   `computable` 留未知）。scaffold 复用（不重算）`candidates_v3/aldehydes_with_scaffold_split.parquet`
   —— 其 `id` 列在全量 220,524 行重叠上逐位校验 == `ald_idx`（0 处不符，raw SMILES
   精确匹配），所以按行序合并是安全的。335 行（不在那个 parquet 里）scaffold 为
   None；0 行 RDKit 标准化失败；0 个 canonical SMILES 重复。脚本：
   `cross_benzoin/build_aldehyde_index.py`。
2. ✅ **2026-09-11 完成 —— 结果是校验，不是重建。** 核对 `homo_v6/aldehydes_all.csv`
   现有的 `id` 列（经 `qc.norm_id`）对 `ald_idx`：**全量**（不是抽样）209,526/209,526
   行 1:1 匹配、两边 0 孤儿 —— `aldehydes_all.csv` 其实早就用 `ald_idx` 正确 key 了，
   只是没写文档。真发现一个问题：它的 `smiles` 列**不可靠地 canonical** —— 2,604/209,526
   （1.24%）跟新算的 `smiles_canonical` 只是表示法不同（比如 Kekulized 大写芳香 vs
   RDKit 小写 canonical 形式 —— 同一分子，抽样 8 个手工核对过）。**所以：以后的缓存要
   join 到 `ald_idx`，不要 join 到 SMILES 字符串相等** —— 这正是冻结索引要消除的漂移。
3. ✅ **2026-09-11 完成。** `cross_benzoin/chemical_space.py` 的
   `FlyingDataset.pair(i,j)`。实现中发现一个需要修正的地方：260 特征里只有一部分
   真是惰性的——donor/acceptor 局部 QM+BDE（醛索引缓存）+ 三个 RDKit-2D 块 +
   `interaction_*` 项确实能从两个醛缓存拼出来；但 `product_*` QM 和
   `product_mordred_*`（查过：用 `ignore_3D=False`，好几个描述符族本质是 3D 的）
   都需要产物自己的优化几何，跟 baseline / label 一样要走 DFT/xTB 流水线，**不是
   惰性的**。所以 `pair(i,j)` 返回一个永远有的 `lazy_features` dict + 只有已算过
   的对才有的 `computed_full`/`known_row`（原样读回，不重算），而不是承诺里那个
   单一的 260 维向量。**校验**（`verify_chemical_space_pair.py`，从 round10 冠军表
   抽 20 个已知对）：两个确定性层——RDKit-2D 600/600、`interaction_*` 全部、
   product_smiles 20/20——**逐位精确匹配**；donor/acceptor QM 在 0.2% 相对容差内
   995/1380 匹配，其余的偏差跟已知的"醛重算保真度"现象一致（round10 表用的 QM
   快照比现在重算过的 `aldehydes_all.csv` 略早，判断标准是相对偏差不是相等）——
   不是 bug。路上还抓到并修了校验脚本自己的一个真 bug：靠搜索匹配 `pair_key` 来
   反推 (donor,acceptor) 地址，在某个 `pair_key` 同时对应两种角色顺序时会静默取
   错行；改成直接用每一行自己的 donor/acceptor SMILES 解析地址。
4. 加标签 + split + 基线。
5. 迁移 35,528 个标签；退役 `candidates_v3`（移走，不删）。
6. 之后才：把重做的 AL / 筛选指向它。

**第 3 步的校验不能跳** —— 特征拼装的静默漂移会毒害之后每一个模型。
