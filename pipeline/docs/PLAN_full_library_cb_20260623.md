# 全库 cb featurization — 点火前评估 (2026-06-23)

## 1. homo_pairs.csv —— 就绪 ✓
`data/cross_benzoin/homo_v6/homo_pairs.csv`：**220,724 行**，列 `donor_id,acceptor_id,donor_smiles,acceptor_smiles`，
正是 `submit_cb_featurize_array.sh` 期望的格式。来源 = screen_v6 中 `xtb_optimized==True` 的分子
（merged CSV `data/raw/screen_v6/analysis/screen_v6_features_mwf_all.csv` 保留，chunk 目录已在配额清理时删除，不影响）。
→ 输入侧无需再做。

## 2. "screen 的 funnel_v3 几何" —— 已澄清，无需重跑 screen
关键认识：**cb_featurize 从 SMILES 重新做 funnel_v3 几何，不复用 screen 的 legacy-ranker 几何。**
- screen（legacy `_rank_conformers`）只是 **T2 triage / SMILES 来源**；homo_pairs 只带 SMILES+id，无几何。
- cb_featurize 拿这些 SMILES **重新跑 funnel_v3**（ETKDG→GFN-FF→GFN2 opt+拓扑守卫）→ 几何与训练表（funnel_v3）一致。
- 所以记忆里说的"screen 需用 funnel_v3 重跑"，真实含义是 **featurization 要用 funnel_v3 重做** = 这次全库 cb 本身。
  **不需要**单独把 triage screen 重跑一遍。screen 的 legacy 几何只服务于已完成的 dG_xtb 粗筛，会被 cb 的 funnel_v3 取代。

## 3. ⚠️ 最大优化点 —— g-xTB SP 必须并入 cb，否则几何算两遍 (~省 7 万核时)
- 生产模型基线是 **g-xTB**（已 ship），所以 220k screen 需要每个分子的 **g-xTB ΔG**。
- 但 **cb_featurize 现在只算 GFN2**（`dG_xtb_kcal`），不含 g-xTB（已确认 grep 无 gxtb）。
- 而 `gxtb_baseline.py::_species()` **会重新做 funnel_v3 + GFN2 ohess + g-xTB SP** —— 与 cb **重复算同一套昂贵几何**。
- 现成本（B 实测 **0.384 核时/对**）：
  | 方案 | 核时 |
  |---|---|
  | cb_featurize 单独 (220,724 对) | **~84,800** |
  | + gxtb_baseline 独立(再做一遍几何) | **再 +~85,000** → 合计 ~170,000 |
  | **融合：cb 内几何上加一次 g-xTB SP** | **~95,000–100,000**（g-xTB SP 仅 +10–20%） |
- **结论：把 g-xTB SP 加进 cb_featurize 的 featurize_aldehyde/featurize_pair**（ohess 几何上一次
  `xtb --gxtb --sp --cosmo dmso`，G_gxtb = E_gxtb + (G_gfn2−E_gfn2)，输出 `dG_gxtb_kcal` 列）。
  代码现成：`gxtb_baseline.py` 的 `_gxtb_sp()` + `_species()` 模式可直接搬。**省 ~7 万核时，且产物只算一遍几何。**

## 4. 耗时 (融合方案, %128)
- 2,207 chunks (CHUNK=100)，每 chunk ~中位 1.6h（B 50 对中位 48min × 2，含 g-xTB +~15%），重 chunk 可达 ~2.5h。
- %128 并发：2207/128 ≈ 17.3 波 × ~1.6h ≈ **~28h 纯计算墙钟**，加排队 → **~1.5–2.5 天**。
- 单 chunk 远在 12h walltime 内（安全）。

## 5. 风险
| 风险 | 评级 | 缓解 |
|---|---|---|
| **核时预算 ~10 万核时**（融合）/ ~17 万（不融合） | 高 | 先融合 g-xTB；先 pilot 验证再全开 |
| **straggler 浪费**：24 核任务等单个 60 原子分子的 ohess(O(N³)) | 中 | 按 heavy-atom 排序分桶 / size-cap 构象数 |
| **inode @%128**：峰值 ~128×12×单分子 scratch ≈ 46 万瞬态 inode | 中 | 已有 per-mol rmtree + MULTIWFN=0 + 节点自愈 + cron；点火前用节点 `mmlsquota` 确认配额余量 |
| **大分子 OOM**：mem=48G/24核=2G/核，60 原子 ohess 可能超 → SIGKILL → 孤儿 | 中 | 调高 mem 或重分子降 workers；cron 兜底孤儿 |
| **产物 SMILES 错误**（[[benzoin-generator-formyl-bug]]）：SMARTS 建产物，220k 下小错率=数千个错产物 | 中 | 点火前跑 `check_smiles.py`（[[smiles-check-workflow-gate]]，纯 RDKit，便宜） |
| homo_pairs 220,724 vs 库 220,859（少 135） | 低 | = xtb_optimized 子集，正常 |

## 6. 由 B 模拟得到的可改进点（按收益排序）
1. **融合 g-xTB SP 进 cb**（#3）—— 省 ~7 万核时，单几何出 GFN2+g-xTB 双基线。**最高优先**。
2. **先 pilot 后全开**：先交 3–5 个 chunk，验证 g-xTB 列正确 + 计时 + 0 quota + SMILES，再放 2207。
   绝不盲开 ~10 万核时。
3. **size-aware 分桶 / size-cap**：按 heavy-atom 数排序，大分子单独小 CHUNK 或限构象数 → 削平墙钟、减核浪费。
4. **预检 SMILES**：对整张 homo_pairs 跑 check_smiles（反应物+产物），剔除/修正错产物后再算。
5. **重 chunk 的 mem/worker 调参**，避免 ohess OOM。
6. （次要）醛阶段加心跳日志（现仅 25/50、50/50 两行，看着像静默）。

## 推荐执行顺序
1. 把 g-xTB SP 融进 cb_featurize（+ `dG_gxtb_kcal` 列）。
2. `check_smiles.py` 预检 homo_pairs；节点上 `mmlsquota` 确认 inode 余量。
3. **Pilot**：`--array=0-4%5` 跑 5 chunk，核对输出列(GFN2+g-xTB dG)、计时、0 quota、SMILES。
4. 全开 `--array=0-2206%128`（MULTIWFN=0, EMIT_ALD=1, CONFORMER=funnel_v3）。
5. 期间保持 orphan_cron 存活；跑完 concat → 接 g-xTB 基线模型 predict 全库。
