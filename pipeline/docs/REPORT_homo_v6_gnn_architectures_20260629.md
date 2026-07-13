# homo_v6 GNN architecture guide (2026-06-29)

A consolidated tour of every GNN tried in the homo_v6 g-xTB→DFT Δ-learning study
(target = DFT − g-xTB from product SMILES). Companion to
`REPORT_MASTER_gxtb_dft_correction_20260625.md` §3, §6, §6d. All numbers on the full
n≈219k dataset, same 70/20/10 (seed 42) split, MLflow exp `exp1_gnn_arch_full` /
`exp1_gnn_dual_full`.

---

## A. Shared backbone (all 2D archs swap only the conv operator)

`gnn_arch_study_gxtb_dft.py:119-154` — identical pipeline so differences are attributable
to the conv operator alone:

```
atom/bond feats → Linear to h → L conv layers (residual x += ReLU(BN(conv))) 
                → readout [global_mean_pool ⊕ global_add_pool] → MLP head → Δ
```
- Node feats: element, degree, formal charge, hybridization, #H, aromatic, in-ring.
- Bond feats: bond-type one-hot + conjugated + in-ring.
- The ONLY variable is the middle `conv` (gine / gat / gcn / nnconv).

## B. The five 2D conv operators

| name | PyG operator | config | uses bond feats? | essence |
|---|---|---|---|---|
| **gine** | `GINEConv` | h128, L4 | yes | edge-aware GIN, sum aggregation, strongest-expressivity baseline |
| **gine_big** | `GINEConv` | h256, L5 | yes | pure capacity test (wider/deeper) |
| **gat** | `GATv2Conv` | h128, 4 heads, edge_dim | yes | **attention**: neighbors unequally weighted, learned per edge |
| **gcn** | `GCNConv` | h128, L4 | **no** | spectral conv, degree-normalized mean; **drops bond feats** (ablation) |
| **nnconv** | `NNConv` | h96, L3 | yes (strong) | **edge-conditioned MPNN**, closest to D-MPNN |

**gat (GATv2Conv)** — instead of fixed-weight aggregation, computes an attention coefficient
α_ij per edge: `h_i' = Σ_j α_ij · W·h_j`, where α from `(h_i, h_j, e_ij)` via a small net +
softmax. Uses GATv2 (dynamic attention), 4 heads (multi-relation), bond feats fed via `edge_dim`.

**gcn (GCNConv)** — the classic/simplest GNN: `h_i' = Σ_j (1/√(d_i d_j))·W·h_j`, symmetric
degree-normalized mean of neighbors. **Deliberate flaw**: GCNConv takes no edge features →
single/double/aromatic bonds are flattened. It's the **ablation baseline** quantifying the value
of bond information. `forward` passes only `(x, edge_index)` for the gcn branch (lines 142-143).

**nnconv (NNConv)** — the heaviest operator: a small net `enet` turns each edge's features into
an h×h weight matrix that transforms the neighbor message: `h_i' = Σ_j NN(e_ij)·h_j`, `aggr="mean"`.
"Bond type decides how information is passed" — the closest PyG analog of D-MPNN/Chemprop. Most
expensive (one matrix per edge), so shrunk to h96, L3.

## C. Results — pure graphs all lose to tabular (n≈219k, same split)

| model | test MAE | reading |
|---|---|---|
| gine | 2.60 | edge-aware baseline |
| gine_big | 2.58 | capacity ≈ no help |
| **gat** | **2.58** | attention brings no edge |
| **nnconv** | 2.62 | heaviest operator still doesn't win |
| **gcn** | 2.68 | worst (lost bond feats → bond info ≈ 0.06–0.1 kcal) |
| gine_hybrid (+34 QM) | 1.99 | only one approaching tabular — via QM, not graph |
| dual_qm (+56 QM) | 1.65 | ties the tabular ensemble |
| — tabular ensemble (ref) | 1.61 | production model |

**Core conclusion: swapping the conv operator (gine↔gat↔nnconv) only wobbles 2.58–2.68 →
architecture is NOT the bottleneck, information is.** Pure 2D graphs plateau ~2.6; only
concatenating QM descriptors at readout breaks it (gine_hybrid 1.99 → dual_qm 1.65), and the
entire gain comes from the QM — gat's attention and nnconv's edge-conditioned matrices extract
nothing extra from topology.

## D. Two information-augmented GNNs

**gine_hybrid** (`hybrid=True`, same script): gine backbone, concatenates 34 product QM
descriptors onto the molecule vector at readout before the head (`hin = 2h + NQM`). Proves QM
injection works (2.6 → 1.99).

**dual_qm dual-encoder** (`gnn_dual_encoder.py`): **product graph + aldehyde graph**, two GINE
encoders combined by the thermodynamic cycle `[h_P, h_A, h_P − 2·h_A]`, then + 56 QM. Key finding:
**adding the aldehyde GRAPH alone barely helps** (2.555 → 2.540; the homo product graph already
contains the aldehyde substructure) — the value is in the aldehyde's QM descriptors, not its
topology. Given all 56 QM, dual_qm hits **1.616 / R²0.84, exactly tying the tabular ensemble**;
remove the QM and it falls back to 2.46 → **graph encoder net contribution ≈ 0**.

## E. 3D GNNs (`gnn3d_schnet_dimenet.py`, 60k subset)

Consume product 3D coordinates (z + pos), no hand-crafted descriptors:
- **SchNet** 2.27 — continuous-filter convolutions, interatomic distances only.
- **DimeNet++** 2.04 — directional message passing with **bond angles** (best 3D; angles beat
  pure distance).
- ViSNet — diverged numerically (untuned).

All lose to the same-data 56-feat MLP (1.83) — hand-crafted descriptors already distill the
dispersion/electronic physics; learning 3D geometry from scratch needs more data / electronic
node features.

## F. One-sentence summary

homo_v6 swept GNNs across the architecture axis: **2D conv operators (gine/gine_big/gat/gcn/nnconv)
all plateau ~2.6** (gcn worst, having dropped bond feats), **3D (SchNet/DimeNet++) ~2.0–2.3**, and
only QM-injection at readout (gine_hybrid 1.99 → dual_qm 1.65) reaches — but never beats — the CPU
tabular ensemble (1.61). The conclusion is identical across all architectures: **the bottleneck is
information (both-side QM descriptors), not network structure** — so production ships the cheap
tabular ensemble, not any GNN.

Figures: `26_gnn_arch_comparison.png` (bar comparison), `27_gnn_best_parity.png`.
