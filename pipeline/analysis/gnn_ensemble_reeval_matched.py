#!/usr/bin/env python
"""Fix-up for gnn_dual_qm_champion275_ensemble.py: that run trained fine (TEST MAE 1.568,
consistent with the original 24478591's 1.552) but its stacking-ensemble comparison was
methodologically broken -- the GNN script's own `np.random.default_rng(42).permutation(N)`
test split and the tabular finalize script's independent split use the SAME seed but a
(slightly) DIFFERENT N (each script builds its population separately, with its own dropna
footprint), so the "last 10%" selected by each permutation is an essentially unrelated random
~10% subset of the shared ~218k pool -- expected overlap by chance is ~10% of one test set,
not the >90% the original script asserted on. Confirmed by the AssertionError in job 24482531.

Fix: reuse the ALREADY-TRAINED model weights (gnn_dual_champion275_state_20260707.pt, saved by
the previous run) and re-evaluate them on the EXACT tabular test-set molecule ids
(test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv) instead of the GNN's own random split.
No retraining needed -- just a forward pass. Reproduces ym/ysd (the delta-target
standardization constants) by rerunning the identical deterministic train-split derivation
(same seed, same N, pure function -- no randomness beyond what's already fixed).
"""
import json, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
RDLogger.DisableLog("rdApp.*")

ROOT = "/scratch-shared/schen3/benzoin-dg"
H = f"{ROOT}/data/cross_benzoin/homo_v6"
DFTDIR = f"{ROOT}/data/raw/dft_sp_funnelv3"
OUT = f"{H}/viz_gxtb_20260625"
CACHE = f"{OUT}/gnn_dual_cache_275champ.pt"
STATE = f"{OUT}/gnn_dual_champion275_state_20260707.pt"
TABULAR_PRED = f"{OUT}/test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

PROD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega",
  "xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH",
  "wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC",
  "fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L",
  "sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp = [f"ald_{c}" for c in ALD]
GKEYS = ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing",
         "nAmide","has_P","has_B","has_S","has_Si","has_halogen"]
GLOB = [f"g_{k}" for k in GKEYS]
FEATS_72 = PROD_QM + ALDp + GLOB
BDE_COLS = ["prod_bdfe_gxtb_kcal", "ald_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]

class PairData(Data):
    def __inc__(self, key, value, *a, **k):
        if key=="edge_index_p": return self.x_p.size(0)
        if key=="edge_index_a": return self.x_a.size(0)
        return super().__inc__(key, value, *a, **k)

def gfeats(smi):
    from rdkit.Chem import rdMolDescriptors, Descriptors
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return {f"g_{k}": np.nan for k in GKEYS}
    s = {a.GetSymbol() for a in m.GetAtoms()}
    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),
            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]
    return {f"g_{k}": v for k, v in zip(GKEYS, vals)}

def add_global(df, smi_col):
    u = df[[smi_col]].drop_duplicates()
    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values
    return df.merge(g, on=smi_col, how="left")

def build_champion_df():
    dft = pd.read_parquet(f"{DFTDIR}/dft_labels_all.parquet", columns=["id", "dG_orca_kcal"]).dropna(
        subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "donor_smiles",
                    "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})

    kept_mordred = set(json.load(open(f"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json"))["kept_mordred"])
    prod_kept = [c for c in kept_mordred if not c.startswith("ald_")]
    ald_kept_raw = [c[len("ald_"):] for c in kept_mordred if c.startswith("ald_")]
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c in prod_kept]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c in ald_kept_raw]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]: prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]: ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")

    FEATS = FEATS_72 + prod_mrd_cols + ald_mrd_cols + BDE_COLS
    df = full.merge(dft, on="id")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal", "smiles", "donor_smiles"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    return df, FEATS

blob = torch.load(CACHE, weights_only=False)
pairs, FEATS = blob["pairs"], blob["feats"]
print("loaded cache", len(pairs), flush=True)
df, _ = build_champion_df()
assert len(df) == len(pairs)
ids = df["id"].astype(int).values
for k, d in enumerate(pairs): d.id = int(ids[k])
print("attached ids", flush=True)

NQM = len(FEATS)
N = len(pairs)
rng = np.random.default_rng(42); idx = rng.permutation(N)
ntr = int(.7 * N); tr = idx[:ntr]
ys = torch.tensor([pairs[i].y.item() for i in tr]); ym, ysd = ys.mean().item(), ys.std().item()
print(f"recovered ym={ym:.4f} ysd={ysd:.4f} from train split (n={len(tr)})", flush=True)
train_ids = set(int(pairs[i].id) for i in tr)  # exclude: molecules the GNN actually backpropped
                                                # on -- keeping them would let memorization
                                                # inflate its apparent accuracy on the overlap

AD = pairs[0].x_p.shape[1]; BD = pairs[0].edge_attr_p.shape[1]

class Enc(nn.Module):
    def __init__(self, h=128, L=4):
        super().__init__(); self.ne = nn.Linear(AD, h); self.ee = nn.Linear(BD, h)
        self.cv = nn.ModuleList(); self.bn = nn.ModuleList()
        for _ in range(L):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h)); self.bn.append(nn.BatchNorm1d(h))
    def forward(self, x, ei, ea, batch):
        x = self.ne(x); e = self.ee(ea)
        for c, b in zip(self.cv, self.bn): x = x + F.relu(b(c(x, ei, e)))
        return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], 1)

class DualGNN(nn.Module):
    def __init__(self, h=128, L=4):
        super().__init__()
        self.encP = Enc(h, L); self.encA = Enc(h, L)
        din = 6 * h + NQM
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, b):
        hP = self.encP(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch)
        hA = self.encA(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        h = torch.cat([hP, hA, hP - 2 * hA, b.qm.view(-1, NQM)], 1)
        return self.head(h).squeeze(-1)

model = DualGNN().to(dev)
model.load_state_dict(torch.load(STATE, map_location=dev))
model.eval()
print("loaded trained weights", flush=True)

tab_ids = set(pd.read_csv(TABULAR_PRED, usecols=["id"])["id"].astype(int).tolist())
match_pairs = [d for d in pairs if d.id in tab_ids and d.id not in train_ids]
n_leaked = sum(1 for d in pairs if d.id in tab_ids and d.id in train_ids)
print(f"tabular test ids={len(tab_ids)} | matched (excl. GNN-train)={len(match_pairs)} | "
      f"excluded as GNN-train-set leakage={n_leaked}", flush=True)
ld = DataLoader(match_pairs, batch_size=256, shuffle=False, follow_batch=["x_p", "x_a"])

ps, gs, ys_raw, ids_ = [], [], [], []
with torch.no_grad():
    for b in ld:
        bb = b.to(dev)
        o = model(bb)
        ps.append(o.cpu()); gs.append(bb.g0.cpu()); ys_raw.append(bb.y.cpu()); ids_.append(bb.id)
pdl = torch.cat(ps).numpy() * ysd + ym
g0 = torch.cat(gs).numpy()
y_delta_true = torch.cat(ys_raw).numpy()
yh = g0 + pdl
yt = g0 + y_delta_true
ids_flat = np.array([i for chunk in ids_ for i in (chunk if isinstance(chunk, list) else chunk.tolist())])
mae_match = float(np.abs(yt - yh).mean())
print(f"GNN re-eval on tabular-matched ids: n={len(ids_flat)} MAE={mae_match:.3f}", flush=True)

gnn_df = pd.DataFrame({"id": ids_flat, "dG_orca_kcal": yt, "gnn_pred": yh})
gnn_df.to_csv(f"{OUT}/gnn_test_predictions_MATCHED_champion275_20260707.csv", index=False)

tab = pd.read_csv(TABULAR_PRED, usecols=["id", "dG_orca_kcal", "dG_pred"]).rename(columns={"dG_pred": "tab_pred"})
merged = gnn_df.merge(tab, on="id", suffixes=("_gnn", "_tab"))
print(f"final merged overlap n={len(merged)}", flush=True)
y_true = merged["dG_orca_kcal_gnn"].values
gnn_p = merged["gnn_pred"].values
tab_p = merged["tab_pred"].values
mae_gnn = float(np.abs(y_true - gnn_p).mean())
mae_tab = float(np.abs(y_true - tab_p).mean())
rows = [{"w_gnn": 0.0, "mae": mae_tab, "config": "tabular_only"}]
best_w, best_mae = 0.0, mae_tab
for w in np.arange(0.05, 1.0, 0.05):
    blend = (1 - w) * tab_p + w * gnn_p
    m = float(np.abs(y_true - blend).mean())
    rows.append({"w_gnn": round(float(w), 2), "mae": m, "config": "blend"})
    if m < best_mae: best_mae, best_w = m, w
rows.append({"w_gnn": 1.0, "mae": mae_gnn, "config": "gnn_only"})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/ensemble_stack_champion275_20260707.csv", index=False)
print(f"overlap n={len(merged)} | tabular-only MAE={mae_tab:.3f} | gnn-only MAE={mae_gnn:.3f} | "
      f"best blend w_gnn={best_w:.2f} MAE={best_mae:.3f} (delta vs tabular {best_mae-mae_tab:+.3f})", flush=True)
print("DONE", flush=True)
