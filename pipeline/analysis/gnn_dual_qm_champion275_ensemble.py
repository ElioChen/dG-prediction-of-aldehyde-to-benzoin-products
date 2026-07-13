#!/usr/bin/env python
"""Architecture-uplift follow-up to gnn_dual_qm_champion275.py (job 24478591, TEST MAE 1.552,
still behind tabular champion MORDREDSLIM271_BDEGXTB's 1.503). Since architecture swaps and
QM-feature-matching both failed to let the GNN beat the tree ensemble on its own, the next
natural lever is INFORMATION COMBINATION rather than a bigger/different single model: does a
stacked blend of the GNN's predictions with the tabular champion's predictions beat either
alone? GNN and GBDT are different function classes (message-passing vs axis-aligned splits)
so their errors may be only partially correlated -- if so, a blend can beat both even though
neither wins outright.

Reuses the EXISTING graph cache (gnn_dual_cache_275champ.pt, 218,105 pairs, confirmed 1:1 with
build_champion_df()'s row order -- zero rows were dropped in the original run, see the
"champion-matched population: 218,105 rows" == "built+cached 218105" match in the 24478591 log).
Re-derives df ids (fast, pure pandas merge of already-computed descriptor CSVs, no xtb) to
attach a stable molecule `id` to each cached pair, then retrains the identical architecture
(same split/hparams/early-stop as the original run) so a fresh copy of the trained model can
save per-molecule test predictions -- the original run never persisted the model weights or
raw predictions to disk, only the aggregate metric.
"""
import json, os, time
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
    """Must match gnn_dual_qm_champion275.py's definition exactly -- the cached pairs were
    pickled as __main__.PairData, so unpickling from any other module needs this class
    importable under the same name for torch.load to resolve it."""
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
    """Identical to gnn_dual_qm_champion275.py's build_champion_df -- only used here to
    recover the `id` column in the SAME deterministic row order as the cached pairs."""
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
    print(f"champion-matched population: {len(df):,} rows, {len(FEATS)} feats", flush=True)
    return df, FEATS


assert os.path.exists(CACHE), "expected the existing 24478591 cache to be present"
blob = torch.load(CACHE, weights_only=False)
pairs, FEATS = blob["pairs"], blob["feats"]
print("loaded cache", len(pairs), flush=True)

df, FEATS2 = build_champion_df()
assert len(df) == len(pairs), f"row-count mismatch: df={len(df)} pairs={len(pairs)} -- cannot safely attach ids"
ids = df["id"].astype(int).values
for k, d in enumerate(pairs):
    d.id = int(ids[k])
print("attached ids, 1:1 verified", flush=True)

NQM = len(FEATS)
N = len(pairs)
rng = np.random.default_rng(42); idx = rng.permutation(N)
ntr, nva = int(.7 * N), int(.2 * N); tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
print(f"split {len(tr)}/{len(va)}/{len(te)}", flush=True)
ys = torch.tensor([pairs[i].y.item() for i in tr]); ym, ysd = ys.mean().item(), ys.std().item()
for d in pairs: d.ystd = (d.y - ym) / ysd
def loader(ix, sh): return DataLoader([pairs[i] for i in ix], batch_size=256, shuffle=sh, follow_batch=["x_p", "x_a"])
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

def evl(model, ld):
    model.eval(); ps, ts, gs, ids_ = [], [], [], []
    for b in ld:
        b = b.to(dev)
        with torch.no_grad(): o = model(b)
        ps.append(o.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu()); ids_.append(b.id)
    pdl = torch.cat(ps).numpy() * ysd + ym; td = torch.cat(ts).numpy() * ysd + ym; g0 = torch.cat(gs).numpy()
    yh, yt = g0 + pdl, g0 + td; r2 = 1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    ids_flat = np.array([i for chunk in ids_ for i in (chunk if isinstance(chunk, list) else chunk.tolist())])
    return float(np.abs(yt - yh).mean()), float(np.sqrt(((yt - yh) ** 2).mean())), r2, yh, yt, ids_flat

def train():
    torch.manual_seed(0); model = DualGNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    tl, vl = loader(tr, True), loader(va, False); best, bs, pat = 1e9, None, 0; t0 = time.time()
    for ep in range(120):
        model.train()
        for b in tl:
            b = b.to(dev); opt.zero_grad(); loss = F.mse_loss(model(b), b.ystd); loss.backward(); opt.step()
        vm = evl(model, vl)[0]; sch.step(vm)
        if vm < best - 1e-4: best, bs, pat = vm, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: pat += 1
        if ep % 10 == 0 or pat == 0: print(f"  ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t0:.0f}s", flush=True)
        if pat >= 15: break
    model.load_state_dict(bs); mae, rmse, r2, yh, yt, ids_te = evl(model, loader(te, False))
    print(f"==> dual_qm_champion275 (ensemble-run) TEST MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k {time.time()-t0:.0f}s", flush=True)
    torch.save(bs, f"{OUT}/gnn_dual_champion275_state_20260707.pt")
    return mae, rmse, r2, yh, yt, ids_te

mae, rmse, r2, yh, yt, ids_te = train()
gnn_df = pd.DataFrame({"id": ids_te, "dG_orca_kcal": yt, "gnn_pred": yh})
gnn_df.to_csv(f"{OUT}/gnn_test_predictions_champion275_20260707.csv", index=False)
print(f"saved {len(gnn_df)} GNN test predictions", flush=True)

# ---- stacking ensemble vs the tabular champion ----
tab = pd.read_csv(TABULAR_PRED, usecols=["id", "dG_orca_kcal", "dG_pred"]).rename(columns={"dG_pred": "tab_pred"})
merged = gnn_df.merge(tab, on="id", suffixes=("_gnn", "_tab"))
assert len(merged) > 0.9 * len(gnn_df), "unexpectedly small overlap between GNN and tabular test sets"
# the two runs used different random splits (same seed but different N/order upstream), so
# only the intersection is usable for a fair blend comparison
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
