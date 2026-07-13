# BDE/BDFE 方法层级 1：GFN2-xTB 原始电子能 BDE

**状态：已采纳（勉强算真实提升，后被 g-xTB 版本取代——见
[METHODS_BDE_gxtb_20260707_zh.md](METHODS_BDE_gxtb_20260707_zh.md)）**

## 动机

醛的 C(=O)-H 键解离能（BDE）是苯偶姻缩合反应中 Breslow 中间体形成时机理上被活化的键；产物的
新生 C-C 键 BDE 则是反应中真正生成的那根键。现有 72 特征冠军集只有 `wbo_CO`（Wiberg 键级）
作为间接的键强度代理，没有真正的解离能。这是 BDE/BDFE 描述符家族中最便宜、最简单的一员。

## 方法

`pipeline/compute/calc_bde.py`。对每个分子：
1. 生成对应的均裂片段（醛 C(=O)-H 自由基 + H 原子；产物新生 C-C 自由基对）。
2. 对母体分子和每个片段分别做一次普通的 xtb `--opt`/单点计算（GFN2-xTB，气相）。
3. `BDE = E_el(片段A) + E_el(片段B) - E_el(母体)` —— 只用原始电子能，**不含热/零点能/熵校正**
   （那是层级 2 的 BDFE 版本）。

不引入新依赖，沿用项目既有的 xtb 调用范式。

## 覆盖率

全库：醛 C-H BDE 220,524/220,524（100%，任务 24394017）；产物新生 C-C BDE 219,021/219,022
（99.8%，任务 24394018 + 重跑——有 2 个 chunk 即使 4 小时超时也命中确定性的
xtb/`rdDetermineBonds` 卡死，接受这约 0.18% 的已知缺口）。附属文件：
`aldehydes_bde_descriptors.csv`、`products_bde_descriptors.csv`。

## 结果

`finalize_correction_bde.py`（任务 24415115），在 mordredslim271 基线（271 特征，MAE 1.612）
上做快速 2 成员 XGB 检验：

| 配置 | 特征数 | 测试集 MAE | 差值 |
|---|---|---|---|
| mordredslim271 基线 | 271 | 1.612 | — |
| + 原始电子能 BDE（两侧：`bde_prod_CC_kcal`、`bde_ald_CH_kcal`） | 273 | 1.588 | **-0.024** |

该差值正好卡在已确立噪声带边缘（0.02-0.03 kcal/mol，见
`REPORT_robustness_baseline72_20260702.md`）——判定为**勉强真实、幅度小**（与更早的
"RDKit-no-glob" 结果属于同一档位）。值得注意的是参数效率很高：只用 2 个额外特征就带来这个信号。

## 为何被取代而非直接保留

GFN2 层级并不是实际被校正的层级（生产基线是 g-xTB）。后续一个假设（"方法不匹配"：在与被校正
对象同一层级上计算 BDE 是否能携带更多信号？）促使在 g-xTB 层级重新推导 BDE，结果确实带来更大
的提升（-0.032 对比本层级的 -0.024）——见
[METHODS_BDE_gxtb_20260707_zh.md](METHODS_BDE_gxtb_20260707_zh.md)。GFN2 层级的原始 BDE 本身
从未被采纳进生产 bundle；它是一个过渡性结果，验证了"原始电子能 BDE 携带真实信号"这一结论，随后
被更好的 g-xTB 版本取代。
