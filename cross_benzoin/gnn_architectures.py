#!/usr/bin/env python
"""
Architecture variants for cross-benzoin's own TripleGNN (train_cross_gnn.py), kept in a
separate file so the production architecture (imported by predict_cross_champion.py and
every existing checkpoint) is never touched.

Motivated by an explicitly queued-but-never-attempted user request (see memory
overnight-handoff-20260717.md's "also queued but not started": "plug an AttentiveFP-style
graph-attention architecture into the existing GNN+tabular stacking framework, as a
comparison point"). The BDE project (pipeline/bde/train_gnn_hybrid_bde.py) already tried an
attentive-pooling variant and found it noise-level under honest scaffold-disjoint evaluation
-- a real prior data point suggesting this may not help here either, but BDE's D-MPNN and
cross's own triple-encoder GINE architecture are different enough (different graph structure,
different pooling baseline: BDE's default was chemprop's mean-aggregation vs this file's
mean+add concat) that a direct test is still warranted rather than assuming the BDE result
transfers.

TripleGNNAttn swaps `Enc`'s `cat(global_mean_pool, global_add_pool)` readout for a learned
gated-attention pooling (torch_geometric.nn.GlobalAttention, the standard "AttentiveFP-style"
soft-attention readout: a per-atom gate score, softmax-normalized within each graph, weighted
sum of atom embeddings) -- same GINEConv message-passing backbone, only the readout differs,
so any effect is attributable to the pooling mechanism specifically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention
from torch_geometric.utils import to_dense_batch


class EncAttn(nn.Module):
    def __init__(self, ad, bd, h=128, layers=4):
        super().__init__()
        self.ne = nn.Linear(ad, h)
        self.ee = nn.Linear(bd, h)
        self.cv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(layers):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h))
            self.bn.append(nn.BatchNorm1d(h))
        gate_nn = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.pool = GlobalAttention(gate_nn)

    def forward(self, x, ei, ea, batch):
        x = self.ne(x)
        e = self.ee(ea)
        for c, b in zip(self.cv, self.bn):
            x = x + F.relu(b(c(x, ei, e)))
        pooled = self.pool(x, batch)
        # keep output dim (2h) identical to the default Enc so TripleGNNAttn's downstream
        # `din = 8*h + nqm` head sizing needs no changes -- duplicate the attention-pooled
        # vector rather than concatenating a second, different pooling (mean/add would defeat
        # the point of isolating attention's effect)
        return torch.cat([pooled, pooled], 1)


class TripleGNNAttn(nn.Module):
    """Same structure as train_cross_gnn.TripleGNN (product+donor+acceptor encoders,
    donor-acceptor asymmetry term, QM-scalar channel) but with EncAttn's attention pooling
    instead of TripleGNN's Enc's mean+add pooling."""

    def __init__(self, ad, bd, nqm, h=128, layers=4):
        super().__init__()
        self.encP = EncAttn(ad, bd, h, layers)
        self.encD = EncAttn(ad, bd, h, layers)
        self.encA = EncAttn(ad, bd, h, layers)
        din = 8 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, b):
        hP = self.encP(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch)
        hD = self.encD(b.x_d, b.edge_index_d, b.edge_attr_d, b.x_d_batch)
        hA = self.encA(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        h = torch.cat([hP, hD, hA, hD - hA, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)


ARCHITECTURES = {"default": None, "attentive": TripleGNNAttn}  # "default" resolved to
# train_cross_gnn.TripleGNN by the training script (avoids a circular import here)


class EncAttn3D(nn.Module):
    """EncAttn (attentive-pooling GINE encoder) with a WIDER edge-embedding input to accept
    2 extra bond-feature dims: [standardized real 3D bond length, has_3d flag]. Only used for
    the PRODUCT encoder (see gnn3d_extract_product_geometry.py's module docstring for why
    donor/acceptor geometry isn't available at scale) -- donor/acceptor keep plain EncAttn.
    DGT-paper-motivated (Zhang & Lapkin, Nat Commun 2026): "bond lengths embedded within bond
    features" is exactly their approach for 3D augmentation at the bond-feature level, the
    single highest-leverage lever in their ablations (43.9% MAE reduction on QM9 HOMO/LUMO
    with precise DFT-derived geometry). This project's product xyz is xTB-funnel-optimised
    (GFN2 level, not full DFT) -- closer to their "MMFF/UFF 3D" condition (~23% of the gain)
    than their "precise DFT 3D" condition, so a smaller but still real effect is the honest
    expectation, not the paper's headline number."""

    def __init__(self, ad, bd_plus3d, h=128, layers=4):
        super().__init__()
        self.ne = nn.Linear(ad, h)
        self.ee = nn.Linear(bd_plus3d, h)
        self.cv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(layers):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h))
            self.bn.append(nn.BatchNorm1d(h))
        gate_nn = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.pool = GlobalAttention(gate_nn)

    def forward(self, x, ei, ea, batch):
        x = self.ne(x)
        e = self.ee(ea)
        for c, b in zip(self.cv, self.bn):
            x = x + F.relu(b(c(x, ei, e)))
        pooled = self.pool(x, batch)
        return torch.cat([pooled, pooled], 1)


class TripleGNN3D(nn.Module):
    """Same as TripleGNNAttn (product+donor+acceptor attentive-pooling GINE encoders,
    donor-acceptor asymmetry term, QM-scalar channel) but the PRODUCT encoder additionally
    consumes real 3D bond-length information (see EncAttn3D). Donor/acceptor encoders are
    untouched plain EncAttn -- isolates the 3D-augmentation effect to the one channel where
    real geometry is actually available at scale."""

    def __init__(self, ad, bd, bd_prod_3d, nqm, h=128, layers=4):
        super().__init__()
        self.encP = EncAttn3D(ad, bd_prod_3d, h, layers)
        self.encD = EncAttn(ad, bd, h, layers)
        self.encA = EncAttn(ad, bd, h, layers)
        din = 8 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, b):
        hP = self.encP(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch)
        hD = self.encD(b.x_d, b.edge_index_d, b.edge_attr_d, b.x_d_batch)
        hA = self.encA(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        h = torch.cat([hP, hD, hA, hD - hA, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)


# ─────────────────────────── Phase 2: real pairwise 3D attention ───────────────────────────
# Phase 1 (EncAttn3D/TripleGNN3D above) only injected real BOND lengths as an extra edge
# feature into GINEConv's local (1-hop) message passing -- gave a real-but-unconfirmed
# signal (bootstrap P=0.7976, see memory gnn3d-dgt-inspired-ablation-20260721.md). User
# correctly diagnosed why: (1) bond-length-only misses non-bonded atom pairs that are
# spatially close but topologically far (exactly the information 3D uniquely provides
# over 2D topology), and (2) GINEConv structurally cannot propagate that information no
# matter what's on the edges, since it only ever looks at direct neighbors. DGT's actual
# 43.9%-MAE-reduction result came from a FULL pairwise atom-atom distance bias inside a
# GLOBAL self-attention mechanism. This section implements that: a distance-biased
# multi-head self-attention block (RBF-expanded pairwise distance -> per-head attention
# bias, masked for padding, standard transformer block otherwise) stacked ON TOP of the
# existing GINEConv topology encoder's atom embeddings -- so the model gets BOTH local
# chemistry (bonds/aromaticity/rings via GINEConv) and global through-space proximity
# (via distance-attention) for the same atoms, closer in spirit to a real graph
# transformer than Phase 1's edge-feature hack.


class DistanceBias(nn.Module):
    """RBF-expand pairwise Euclidean distance, project to a per-attention-head scalar
    bias -- same purpose as DGT paper's spherical-harmonics angle encoding / SchNet's
    continuous-filter distance expansion, simplified to plain Gaussian RBF since we only
    need distance (not angle) here."""

    def __init__(self, n_heads: int, n_rbf: int = 16, cutoff: float = 12.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = cutoff / n_rbf
        self.mlp = nn.Sequential(nn.Linear(n_rbf, n_rbf), nn.ReLU(), nn.Linear(n_rbf, n_heads))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (B, N, N) -> (B, n_heads, N, N)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))
        return self.mlp(rbf).permute(0, 3, 1, 2)


class DistanceAttentionLayer(nn.Module):
    """Standard pre-LN transformer block: multi-head self-attention with a distance-based
    additive bias on the attention logits, masked so padding atoms (from variable-size
    to_dense_batch padding) never attend / are never attended to."""

    def __init__(self, h: int, n_heads: int = 4):
        super().__init__()
        assert h % n_heads == 0
        self.h, self.n_heads, self.dh = h, n_heads, h // n_heads
        self.q = nn.Linear(h, h)
        self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h)
        self.o = nn.Linear(h, h)
        self.dist_bias = DistanceBias(n_heads)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Linear(2 * h, h))

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B,N,h), dist: (B,N,N), mask: (B,N) bool, True = real atom
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        k = self.k(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        v = self.v(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5) + self.dist_bias(dist)
        pad = ~(mask.unsqueeze(1) & mask.unsqueeze(2))  # (B,N,N) True = invalid pair
        scores = scores.masked_fill(pad.unsqueeze(1), float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = torch.nan_to_num(attn)  # fully-padded rows -> all -inf -> softmax NaN -> 0
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.h)
        x = self.ln1(x + self.o(out))
        x = self.ln2(x + self.ffn(x))
        return x


class EncAttn3DGlobal(nn.Module):
    """GINEConv topology encoder (local chemistry) followed by `n_dist_layers` distance-
    biased global self-attention layers (through-space proximity) on the SAME atom
    embeddings, then attentive pooling. Requires real 3D coordinates -- product-only, same
    scope limitation as EncAttn3D (donor/acceptor geometry unavailable at scale, see
    [[xyz-geometry-retention-audit-20260721]])."""

    def __init__(self, ad, bd, h=128, layers=4, n_dist_layers=2, n_heads=4):
        super().__init__()
        self.ne = nn.Linear(ad, h)
        self.ee = nn.Linear(bd, h)
        self.cv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(layers):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h))
            self.bn.append(nn.BatchNorm1d(h))
        self.dist_layers = nn.ModuleList([DistanceAttentionLayer(h, n_heads) for _ in range(n_dist_layers)])
        gate_nn = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.pool = GlobalAttention(gate_nn)

    def forward(self, x, ei, ea, batch, pos):
        x = self.ne(x)
        e = self.ee(ea)
        for c, b in zip(self.cv, self.bn):
            x = x + F.relu(b(c(x, ei, e)))
        # global distance-attention refinement on top of the topology-aware embeddings
        x_dense, mask = to_dense_batch(x, batch)          # (B,N,h), (B,N)
        pos_dense, _ = to_dense_batch(pos, batch)          # (B,N,3)
        dist = torch.cdist(pos_dense, pos_dense)            # (B,N,N)
        for layer in self.dist_layers:
            x_dense = layer(x_dense, dist, mask)
        x = x_dense[mask]  # back to sparse (total_atoms_in_batch, h), same row order as `batch`
        pooled = self.pool(x, batch)
        return torch.cat([pooled, pooled], 1)


class TripleGNNDistAttn(nn.Module):
    """Same structure as TripleGNNAttn (product+donor+acceptor, donor-acceptor asymmetry
    term, QM-scalar channel) but the PRODUCT encoder is EncAttn3DGlobal (GINEConv topology
    + distance-biased global self-attention). Donor/acceptor stay plain EncAttn."""

    def __init__(self, ad, bd, nqm, h=128, layers=4, n_dist_layers=2, n_heads=4):
        super().__init__()
        self.encP = EncAttn3DGlobal(ad, bd, h, layers, n_dist_layers, n_heads)
        self.encD = EncAttn(ad, bd, h, layers)
        self.encA = EncAttn(ad, bd, h, layers)
        din = 8 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, b):
        hP = self.encP(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch, b.x_p_pos)
        hD = self.encD(b.x_d, b.edge_index_d, b.edge_attr_d, b.x_d_batch)
        hA = self.encA(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        h = torch.cat([hP, hD, hA, hD - hA, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)
