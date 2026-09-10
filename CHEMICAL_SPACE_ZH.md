# 化学空间 & Flying Dataset —— 规范

> 状态：**规范，未实现。** 2026-09-10 按用户要求写（"cross 的建库之前根本不对，真正的
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
| **基础库** | `data/library/aldehydes_clean_v6.csv` —— **220,860** 行（单醛，过滤后，带 `cho_class` + `xtb_risk`）。设计上包含算不出来的分子（见 `PROJECT_PLAN_ZH.md` §1.2）。 |
| **homo 空间** | 220,860 个对角对 `(i, i)`。 |
| **cross 空间** | 所有有向 `(i, j)`：**220,860² = 4.877 × 10¹⁰** 有向对（488 亿）。无向 ≈ 2.44 × 10¹⁰。 |
| **已落地** | ~35,528 对有 DFT 标签（已暂停的 AL 1–10 轮）。其余全是虚拟的。 |

任何文件都不会存全部对。一个对是一个**地址**，按需计算。

---

## 3. 醛的规范索引（必须先冻结）

- **`ald_idx`** ∈ `[0, 220859]` —— 稳定整数，一次性分配，取自 `aldehydes_clean_v6.csv`
  一份**冻结**副本的行序。永不重派，永不是对过滤视图的 `enumerate()`。
- 一起带：canonical SMILES（RDKit canonical，单一固定协议）、`InChIKey`（人类可读次级
  键）、`cho_class`、`xtb_risk`、Bemis–Murcko scaffold SMILES、`computable` 标志
  （一旦确认某分子几何/DFT 失败就置 false，下游直接跳过不再重试）。
- 产物：`data/chemical_space/aldehyde_index.parquet`（待建）。这个文件是唯一真值源；
  `homo_v6/*` 缓存要重新 key 到它。

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

space.n_aldehydes                      # 220860
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

1. 从 `aldehydes_clean_v6.csv` 冻结 `aldehyde_index.parquet`（派 `ald_idx`、canonical
   SMILES、scaffold、`cho_class`、`xtb_risk`；`computable` 初始未知）。
2. 把一个现有缓存（`aldehydes_all.csv`）重新 key 到 `ald_idx`，对几个已知对做往返校验。
3. 先写 `chemical_space.py` 的 `pair(i, j)` 特征路径；校验
   `space.pair(i, j).features` 对 ~20 个已知对**逐位复现**当前冠军训练表的一行
   （尽可能 bit 级）。
4. 加标签 + split + 基线。
5. 迁移 35,528 个标签；退役 `candidates_v3`（移走，不删）。
6. 之后才：把重做的 AL / 筛选指向它。

**第 3 步的校验不能跳** —— 特征拼装的静默漂移会毒害之后每一个模型。
