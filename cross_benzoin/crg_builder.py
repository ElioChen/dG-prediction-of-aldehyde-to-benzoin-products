#!/usr/bin/env python
"""Condensed reaction graph (CRG) builder for the benzoin coupling, user-requested
2026-09-16 ("dG预测需要尝试使用反应图CRG等进行预测").

Key realization that makes this tractable without an external reaction atom-mapper
(RXNMapper etc.): benzoin coupling (2 RCHO -> R-CO-CH(OH)-R') is atom-economical --
no atoms are lost, so the PRODUCT's own molecular graph already physically contains
every atom of both donor and acceptor, unmodified, as intact substituent trees. A
condensed reaction graph here does not require unioning three separate mol objects;
it only requires (a) locating the 5-atom reactive core within the product's own
SMILES (ketC/ketO/carbC/hydO, no separate hydH node since this project's graphs are
implicit-H, matching `train_cross_gnn.py`'s `graph()`), and (b) tagging every other
atom by which side of the new ketC-carbC bond it sits on. Both are derivable purely
from the product graph's own connectivity -- donor_smiles/acceptor_smiles are not
needed to build the CRG itself (only the QM x_d channel still uses them, unchanged).

Formal net atom mapping (mass-balance-consistent, not a claim about the NHC
Umpolung mechanism's actual intermediates):
  donor CHO_C  -> product ketC   (C=O bond unchanged)
  donor CHO_O  -> product ketO   (unchanged)
  donor CHO_H  -> migrates to become the new hydroxyl's H (implicit, on hydO)
  acceptor CHO_C -> product carbC  (C=O order 2->1: "order-changed")
  acceptor CHO_O -> product hydO   (gains a bond to H: C=O -> C-OH)
  acceptor CHO_H -> stays a plain (implicit) H on carbC, unchanged
  NEW bond: ketC-carbC (does not exist in either separate reactant) -- the
  defining feature this CRG marks explicitly, rather than leaving the model to
  infer "these two disconnected graphs are secretly joined" the way the existing
  3-encoder TripleGNN (`train_cross_gnn.py`) does with 3 separate, disconnected
  graphs (see [[gnn-architectures]] for that design). CC_new is exactly the bond
  the champion schema's own `wbo_CC_new` feature already treats as special.

Reactive-core SMARTS: "[CX3](=O)[CX4][OX2H1]" (match order = ketC, ketO, carbC,
hydO by construction of the pattern's atom order). Validated on 500 random rows
of the champion training table: 488/500 (97.6%) unique match, 3/500 (0.6%)
ambiguous (a look-alike alpha-hydroxy-ketone-like motif elsewhere in a complex
substituent), 9/500 (1.8%) no match (unusual tautomer/protonation SMILES).
Ambiguous/no-match rows return None and are dropped by the caller (same
graceful-drop convention as `assemble_homo_standalone_table.py`'s
`n_ald_missing` handling), not silently mis-mapped.
"""
from __future__ import annotations

from collections import deque

from rdkit import Chem

CORE_SMARTS = Chem.MolFromSmarts("[CX3](=O)[CX4][OX2H1]")

ELEMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]


def _oh(x, ch):
    return [int(x == c) for c in ch] + [int(x not in ch)]


def _atom_feats(a):
    """Same base atom features as train_cross_gnn.py's af(), so a CRG model and
    the disconnected-3-graph champion GNN see identically-encoded chemistry --
    any MAE difference is attributable to the representation (connected +
    reaction-tagged vs disconnected), not to a feature-set change."""
    return (_oh(a.GetSymbol(), ELEMS) + _oh(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _oh(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + _oh(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
            + _oh(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + [int(a.GetIsAromatic()), int(a.IsInRing())])


BTD = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}


def _bond_feats(b):
    v = [0, 0, 0, 0]
    v[BTD.get(b.GetBondType(), 0)] = 1
    return v + [int(b.GetIsConjugated()), int(b.IsInRing())]


def find_reactive_core(mol) -> tuple[int, int, int, int] | None:
    """(ketC, ketO, carbC, hydO) 0-based product-mol atom indices, or None if the
    motif isn't found or is ambiguous (>1 match)."""
    matches = mol.GetSubstructMatches(CORE_SMARTS)
    if len(matches) != 1:
        return None
    return matches[0]  # (ketC, ketO, carbC, hydO) by SMARTS atom order


def side_labels(mol, ket_c: int, carb_c: int) -> list[int] | None:
    """BFS-split every atom into donor-side (0) / acceptor-side (1) by cutting
    the ketC-carbC bond. Returns None if that cut does not disconnect the graph
    (an unexpected extra path between the two sides -- e.g. a macrocycle linking
    donor's and acceptor's R groups directly; not expected for this dataset's
    independent small-molecule aldehydes, but checked rather than assumed)."""
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if {i, j} == {ket_c, carb_c}:
            continue
        adj[i].append(j)
        adj[j].append(i)

    def bfs(start):
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    donor_side = bfs(ket_c)
    acceptor_side = bfs(carb_c)
    if donor_side & acceptor_side:
        return None  # cutting ketC-carbC did not disconnect the graph
    if len(donor_side) + len(acceptor_side) != n:
        return None  # unreachable atoms (shouldn't happen for a connected mol)
    labels = [0] * n
    for i in acceptor_side:
        labels[i] = 1
    return labels


def build_crg(smiles: str):
    """Returns (x, edge_index_pairs, edge_attr, core) or None.
    x: list[list[float]] per-atom [base_feats(20) + side_onehot(2) + is_core(1)]
    edge_index_pairs: list[(i, j)] directed both ways per bond
    edge_attr: list[list[float]] per directed edge [base_feats(6) + is_new_bond(1)]
    core: (ketC, ketO, carbC, hydO) for downstream sanity checks / feature reuse
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    core = find_reactive_core(mol)
    if core is None:
        return None
    ket_c, ket_o, carb_c, hyd_o = core
    labels = side_labels(mol, ket_c, carb_c)
    if labels is None:
        return None
    core_atoms = {ket_c, ket_o, carb_c, hyd_o}

    x = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        is_core = int(i in core_atoms)
        side_oh = [0, 0] if is_core else [1 - labels[i], labels[i]]  # [donor_side, acceptor_side]
        x.append(_atom_feats(a) + side_oh + [is_core])

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        is_new = int({i, j} == {ket_c, carb_c})
        f = _bond_feats(b) + [is_new]
        edge_index += [(i, j), (j, i)]
        edge_attr += [f, f]

    if not edge_index:
        return None
    return x, edge_index, edge_attr, core


def build_crg_delta(smiles: str):
    """Extended CGR with an explicit before/after bond-ORDER encoding on the two
    reaction-core edges, not just a single is_new_bond flag -- the refinement
    `build_crg()`'s own docstring (and LAB_JOURNAL 2026-09-16) flagged as noted
    but not built, done 2026-09-17 on user request ("尝试其他的基于反应的GNN").
    This is the proper "dynamic bond" formalism a Condensed Graph of Reaction
    is normally defined by (bond order before -> bond order after), rather than
    build_crg()'s single new/old flag.

    Per this module's net atom mapping, exactly TWO edges change bond order
    across the reaction (every other edge is a spectator bond, unchanged inside
    donor's or acceptor's own substituent tree, since the coupling only touches
    the 5-atom reactive core):
      ketC-carbC : NEW bond,        before=none   -> after=single (order 0->1)
      carbC-hydO : order-CHANGED,   before=double -> after=single (order 2->1;
                   acceptor's CHO_C=CHO_O becomes carbC-hydO's C-OH)

    Returns (x, edge_index_pairs, edge_attr, core) or None. Node features are
    identical to build_crg() (base_feats(20) + side_onehot(2) + is_core(1)).
    edge_attr grows from build_crg()'s 7 dims to 12:
      base_feats(6, the bond's "after" state: type onehot(4) + conjugated +
      in_ring) + order_before_onehot(4, all-zero means "no bond before") +
      is_new_bond(1) + is_order_changed(1)
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    core = find_reactive_core(mol)
    if core is None:
        return None
    ket_c, ket_o, carb_c, hyd_o = core
    labels = side_labels(mol, ket_c, carb_c)
    if labels is None:
        return None
    core_atoms = {ket_c, ket_o, carb_c, hyd_o}

    x = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        is_core = int(i in core_atoms)
        side_oh = [0, 0] if is_core else [1 - labels[i], labels[i]]  # [donor_side, acceptor_side]
        x.append(_atom_feats(a) + side_oh + [is_core])

    new_bond_pair = {ket_c, carb_c}
    changed_bond_pair = {carb_c, hyd_o}
    double_onehot = [0, 0, 0, 0]
    double_onehot[BTD[Chem.BondType.DOUBLE]] = 1

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pair = {i, j}
        after = _bond_feats(b)
        is_new = int(pair == new_bond_pair)
        is_changed = int(pair == changed_bond_pair)
        if is_new:
            before_onehot = [0, 0, 0, 0]  # no bond existed pre-reaction
        elif is_changed:
            before_onehot = double_onehot  # acceptor's CHO C=O
        else:
            before_onehot = after[:4]  # spectator bond, unchanged
        f = after + before_onehot + [is_new, is_changed]
        edge_index += [(i, j), (j, i)]
        edge_attr += [f, f]

    if not edge_index:
        return None
    return x, edge_index, edge_attr, core


if __name__ == "__main__":
    import sys
    import random
    import pandas as pd

    table = sys.argv[1] if len(sys.argv) > 1 else (
        "data/cross_benzoin/cross_round10/"
        "cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet")
    df = pd.read_parquet(table, columns=["smiles", "donor_smiles", "acceptor_smiles"])
    random.seed(0)
    idxs = random.sample(range(len(df)), min(2000, len(df)))
    ok = fail = 0
    delta_ok = delta_fail = 0
    edge_dim_seen = node_dim_seen = None
    for i in idxs:
        r = build_crg(df["smiles"].iloc[i])
        if r is None:
            fail += 1
        else:
            ok += 1
        rd = build_crg_delta(df["smiles"].iloc[i])
        if rd is None:
            delta_fail += 1
        else:
            delta_ok += 1
            node_dim_seen, edge_dim_seen = len(rd[0][0]), len(rd[2][0])
    print(f"build_crg: ok={ok} fail={fail} / {len(idxs)} ({ok / len(idxs):.1%} usable)")
    print(f"build_crg_delta: ok={delta_ok} fail={delta_fail} / {len(idxs)} "
          f"({delta_ok / len(idxs):.1%} usable), node_dim={node_dim_seen} edge_dim={edge_dim_seen}")
