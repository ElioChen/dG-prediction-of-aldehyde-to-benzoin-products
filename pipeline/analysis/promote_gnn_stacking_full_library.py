#!/usr/bin/env python
"""Promote the CONFIRMED GNN+tabular stacking gain (job 24578348, aligned-v2,
test MAE 1.427 vs tabular-only 1.485, see REPORT_gnn_stacking_ALIGNEDV2_20260713.md)
to a full-library scoring pass.

Loads the saved ALIGNEDV2 GNN checkpoint + the same graph cache it was trained on
(218,105/218,227 molecules -- essentially the whole library already), runs inference
over ALL cached pairs (not just its held-out test slice), and blends with the
existing tabular champion's full-library predictions
(products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv) at w_gnn=0.40 (the
empirically-best weight found in the aligned-v2 report).

Reuses build_tabular_df()/build_gnn_df() verbatim from gnn_dual_qm_champion275_
aligned_v2.py so the tab_train id set -- and therefore the y-standardization
(ym, ysd) the saved model weights were actually trained against -- is reproduced
exactly. No retraining; this is pure inference + blending.
"""
import json, time
from pathlib import Path
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
GNN_STATE = f"{OUT}/gnn_dual_champion275_ALIGNEDV2_state_20260713.pt"
TAB_FULL_LIB = f"{OUT}/products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv"
TAG = time.strftime("%Y%m%d")
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

def _load_mrd_bde():
    kept_mordred = set(json.load(open(f"{OUT}/mordred_slim_selection_20260703.json"))["kept_mordred"])
    prod_kept = [c for c in kept_mordred if not c.startswith("ald_")]
    ald_kept_raw = [c[len("ald_"):] for c in kept_mordred if c.startswith("ald_")]
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c in prod_kept]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False).drop_duplicates("id", keep="first")
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c in ald_kept_raw]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False).drop_duplicates("id", keep="first")
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")
    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).drop_duplicates("id", keep="first").rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).drop_duplicates("id", keep="first").rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]: prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]: ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan
    return prod_mrd, prod_mrd_cols, ald_mrd, ald_mrd_cols, prod_bde, ald_bde

def build_gnn_df():
    dft = pd.read_parquet(f"{DFTDIR}/dft_labels_all.parquet", columns=["id", "dG_orca_kcal"]).dropna(
        subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "donor_smiles",
                    "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    p = p.dropna(subset=["donor_id", "smiles", "dG_gxtb_kcal"]).drop_duplicates("id", keep="first")
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
    prod_mrd, prod_mrd_cols, ald_mrd, ald_mrd_cols, prod_bde, ald_bde = _load_mrd_bde()
    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")
    df = full.merge(dft, on="id")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal", "smiles", "donor_smiles"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    return df

def build_tabular_df():
    dft = pd.read_parquet(f"{DFTDIR}/dft_labels_all.parquet", columns=["id", "dG_orca_kcal"]).dropna(
        subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    p = p.dropna(subset=["donor_id", "smiles", "dG_gxtb_kcal"]).drop_duplicates("id", keep="first")
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
    prod_mrd, prod_mrd_cols, ald_mrd, ald_mrd_cols, prod_bde, ald_bde = _load_mrd_bde()
    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")
    df = full.merge(dft, on="id")
    FEATS = FEATS_72 + prod_mrd_cols + ald_mrd_cols + BDE_COLS
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    return df

# ── reproduce the exact tab_train split aligned_v2 trained the saved GNN against ──
tdf = build_tabular_df()
print(f"tabular population: {len(tdf):,} rows", flush=True)
rng = np.random.default_rng(42); idx = rng.permutation(len(tdf))
ntr = int(.7 * len(tdf))
tab_train_ids = set(tdf["id"].astype(int).values[idx[:ntr]].tolist())
print(f"tab_train = {len(tab_train_ids):,}", flush=True)

# ── load cache + attach ids (same order as build_gnn_df(), matching how the cache was built) ──
assert Path(CACHE).exists()
blob = torch.load(CACHE, weights_only=False)
pairs, GNN_FEATS = blob["pairs"], blob["feats"]
gdf = build_gnn_df()
assert len(gdf) == len(pairs), f"row-count mismatch: gdf={len(gdf)} pairs={len(pairs)}"
gids = gdf["id"].astype(int).values
for k, d in enumerate(pairs): d.id = int(gids[k])
print(f"loaded cache: {len(pairs):,} pairs, ids attached", flush=True)

tr_pairs = [d for d in pairs if d.id in tab_train_ids]
ys = torch.tensor([d.y.item() for d in tr_pairs]); ym, ysd = ys.mean().item(), ys.std().item()
print(f"reproduced ym={ym:.4f} ysd={ysd:.4f} from {len(tr_pairs):,} train pairs", flush=True)

# ── model (identical arch to aligned_v2.py) ──
NQM = len(GNN_FEATS)
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
model.load_state_dict(torch.load(GNN_STATE, map_location=dev, weights_only=False))
model.eval()
print(f"loaded GNN state from {GNN_STATE}", flush=True)

def loader(ps): return DataLoader(ps, batch_size=512, shuffle=False, follow_batch=["x_p", "x_a"])

all_ids, all_pred, all_true, all_g0 = [], [], [], []
t0 = time.time()
with torch.no_grad():
    for b in loader(pairs):
        b = b.to(dev)
        o = model(b).cpu().numpy()
        all_pred.append(o * ysd + ym)
        all_true.append(b.y.cpu().numpy())
        all_g0.append(b.g0.cpu().numpy())
        all_ids.append(b.id.cpu().numpy() if torch.is_tensor(b.id) else np.array(b.id))
pred_delta = np.concatenate(all_pred); true_delta = np.concatenate(all_true); g0 = np.concatenate(all_g0)
ids = np.concatenate(all_ids)
gnn_pred_full = g0 + pred_delta
true_y_absolute = g0 + true_delta  # cache stores y as the RAW DELTA (dG_orca - dG_gxtb), not absolute dG_orca
print(f"GNN full-library inference done: n={len(ids):,} ({time.time()-t0:.0f}s)", flush=True)

gnn_df = pd.DataFrame({"id": ids, "dG_orca_kcal": true_y_absolute, "gnn_pred": gnn_pred_full})

# ── merge with tabular full-library predictions, blend at w_gnn=0.40 (aligned-v2's best) ──
tab_full = pd.read_csv(TAB_FULL_LIB, usecols=["id", "smiles", "dG_gxtb_kcal", "dG_gxtb_corrected_final",
                                               "uncertainty_pi_width", "route_to_dft"])
merged = tab_full.merge(gnn_df, on="id", how="left")
have_gnn = merged["gnn_pred"].notna()
print(f"full library n={len(merged):,}, GNN coverage n={have_gnn.sum():,} ({have_gnn.mean()*100:.1f}%)", flush=True)

W = 0.40
merged["dG_blend_final"] = merged["dG_gxtb_corrected_final"]
merged.loc[have_gnn, "dG_blend_final"] = (1 - W) * merged.loc[have_gnn, "dG_gxtb_corrected_final"] + W * merged.loc[have_gnn, "gnn_pred"]
merged["blend_source"] = np.where(have_gnn, "tabular+gnn_blend", "tabular_only_no_gnn_cache")

out_csv = f"{OUT}/products_dG_corrected_GNNSTACK_w{int(W*100)}_{TAG}.csv"
merged.to_csv(out_csv, index=False)
print(f"wrote {out_csv}", flush=True)

# accuracy check where we have real DFT labels (dG_orca_kcal from the GNN merge, i.e. the labeled subset)
labeled = merged[have_gnn].dropna(subset=["dG_orca_kcal"])
mae_tab = float(np.abs(labeled["dG_gxtb_corrected_final"] - labeled["dG_orca_kcal"]).mean())
mae_blend = float(np.abs(labeled["dG_blend_final"] - labeled["dG_orca_kcal"]).mean())
mae_gnn = float(np.abs(labeled["gnn_pred"] - labeled["dG_orca_kcal"]).mean())
print(f"sanity on full labeled overlap (n={len(labeled):,}, INCLUDES train -- optimistic, not a test-set number): "
      f"tab_MAE={mae_tab:.3f} gnn_MAE={mae_gnn:.3f} blend_MAE={mae_blend:.3f}", flush=True)

with open(f"{OUT}/REPORT_promote_gnn_stacking_full_library_{TAG}.md", "w") as f:
    f.write(f"""# GNN+tabular stacking promoted to full-library scoring ({TAG})

Follows the CONFIRMED aligned-v2 result (job 24578348): full held-out test MAE
tabular-only 1.485 -> blend (w_gnn=0.40) 1.427 (delta -0.058).

This run applies that same w_gnn=0.40 blend to the WHOLE library using the saved
GNN checkpoint (`gnn_dual_champion275_ALIGNEDV2_state_20260713.pt`) and the
existing tabular champion's full-library predictions
(`products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv`).

- Full library n = {len(merged):,}
- GNN cache coverage: n = {have_gnn.sum():,} ({have_gnn.mean()*100:.1f}%) -- the
  ~{len(merged)-have_gnn.sum():,} without a cached graph pair keep the
  tabular-only prediction (`blend_source = tabular_only_no_gnn_cache`).
- Output: `{Path(out_csv).name}` (`dG_blend_final` is the new best-estimate column;
  `dG_gxtb_corrected_final` from the tabular champion kept alongside for reference).
- Sanity check on the full labeled overlap (n={len(labeled):,}, note this INCLUDES
  the GNN's own training rows so it is optimistic, not a held-out number -- the
  trustworthy number is the aligned-v2 test-set MAE above): tabular {mae_tab:.3f},
  GNN {mae_gnn:.3f}, blend {mae_blend:.3f}.

**Recommendation:** use `{Path(out_csv).name}`'s `dG_blend_final` as the new
production prediction column going forward.
""")
print("wrote report", flush=True)
print("DONE", flush=True)
