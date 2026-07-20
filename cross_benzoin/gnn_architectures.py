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
