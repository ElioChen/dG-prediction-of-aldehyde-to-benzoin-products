#!/usr/bin/env python
"""Dual-encoder Delta-GNN (product graph + aldehyde graph, thermodynamic-cycle combine,
architecture proven best in gnn_dual_encoder.py / REPORT_homo_v6_gnn_architectures_20260629.md)
re-run with the FULL current tabular-champion feature set (275 feats: 72 QM + 199 SHAP-pruned
mordred + 4 g-xTB BDE/BDFE) concatenated at readout, instead of the original 56 QM-only feats.

Prior sweep concluded architecture is NOT the bottleneck (all 2D conv operators plateau ~2.6;
only QM-injection breaks the plateau) and dual_qm+56feat (1.65) ties the OLD tabular ensemble
(1.61) but never beat it. This tests whether feeding the GNN the SAME richer 275-feat set the
current tabular champion (MORDREDSLIM271_BDEGXTB, test MAE 1.503) uses lets it match/beat that
number, using the same 70:20:10 (seed 42) split protocol and the same ~219k labeled population
(same dropna criteria: core 72 QM + labels complete, mordred/BDE sparse blocks median-imputed
on train only, matching finalize_correction_mordredslim271_bdegxtb.py's approach).

Only trains dual_qm (product_only/dual-without-QM already characterized at 2.58/2.57 in the
prior full sweep, job 24224245 -- not worth re-running to save GPU time).
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
os.makedirs(OUT, exist_ok=True)
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

ELEMS = ["B","C","N","O","F","Si","P","S","Cl","Se","Br","I"]
def oh(x, ch): return [int(x == c) for c in ch] + [int(x not in ch)]
def af(a):
    return (oh(a.GetSymbol(),ELEMS)+oh(a.GetTotalDegree(),[0,1,2,3,4,5])+oh(a.GetFormalCharge(),[-2,-1,0,1,2])
            +oh(str(a.GetHybridization()),["SP","SP2","SP3","SP3D","SP3D2"])+oh(a.GetTotalNumHs(),[0,1,2,3,4])
            +[int(a.GetIsAromatic()),int(a.IsInRing())])
BTD={Chem.BondType.SINGLE:0,Chem.BondType.DOUBLE:1,Chem.BondType.TRIPLE:2,Chem.BondType.AROMATIC:3}
def bf(b):
    v=[0,0,0,0]; v[BTD.get(b.GetBondType(),0)]=1; return v+[int(b.GetIsConjugated()),int(b.IsInRing())]
def graph(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumAtoms()==0: return None
    x=torch.tensor([af(a) for a in m.GetAtoms()],dtype=torch.float)
    ei,ea=[],[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); f=bf(b); ei+=[[i,j],[j,i]]; ea+=[f,f]
    if not ei: ei=[[0,0]]; ea=[[0,0,0,0,0,0]]
    return x, torch.tensor(ei).t().contiguous(), torch.tensor(ea,dtype=torch.float)

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
    """Mirrors finalize_correction_mordredslim271_bdegxtb.py's data assembly exactly, so this
    trains/evaluates on the SAME ~219k-molecule population with the SAME 275 feature columns."""
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


if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False)
    pairs, FEATS = blob["pairs"], blob["feats"]
    print("loaded cache", len(pairs), flush=True)
else:
    df, FEATS = build_champion_df()
    NQM = len(FEATS)
    # median-impute the sparse mordred/BDE block using a 70% split fit on train rows only
    # (same split protocol as the tabular champion, applied here BEFORE graph building so
    # the imputer only ever sees train statistics)
    rng0 = np.random.default_rng(42); idx0 = rng0.permutation(len(df))
    ntr0 = int(.7 * len(df)); tr0 = idx0[:ntr0]
    med = df[FEATS].iloc[tr0].median()
    Xz = df[FEATS].fillna(med)
    qm_mean = Xz.mean().values; qm_std = Xz.std().replace(0, 1).values
    qmz = (Xz.values - qm_mean) / qm_std
    print("building", len(df), "pairs", flush=True); t0 = time.time(); pairs = []
    for k in range(len(df)):
        gp = graph(df.smiles.iloc[k]); ga = graph(df.donor_smiles.iloc[k])
        if gp is None or ga is None: continue
        d = PairData(); d.x_p, d.edge_index_p, d.edge_attr_p = gp; d.x_a, d.edge_index_a, d.edge_attr_a = ga
        d.y = torch.tensor([float(df.dG_orca_kcal.iloc[k]) - float(df.dG_gxtb_kcal.iloc[k])], dtype=torch.float)
        d.g0 = torch.tensor([float(df.dG_gxtb_kcal.iloc[k])], dtype=torch.float)
        d.qm = torch.tensor(qmz[k], dtype=torch.float).view(1, -1)
        pairs.append(d)
        if k % 20000 == 0: print(f"  {k} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"pairs": pairs, "feats": FEATS, "qm_mean": qm_mean, "qm_std": qm_std}, CACHE)
    print("built+cached", len(pairs), flush=True)

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
    model.eval(); ps, ts, gs = [], [], []
    for b in ld:
        b = b.to(dev)
        with torch.no_grad(): o = model(b)
        ps.append(o.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu())
    pdl = torch.cat(ps).numpy() * ysd + ym; td = torch.cat(ts).numpy() * ysd + ym; g0 = torch.cat(gs).numpy()
    yh, yt = g0 + pdl, g0 + td; r2 = 1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    return float(np.abs(yt - yh).mean()), float(np.sqrt(((yt - yh) ** 2).mean())), r2, yh, yt

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
    model.load_state_dict(bs); mae, rmse, r2, yh, yt = evl(model, loader(te, False))
    print(f"==> dual_qm_champion275 TEST MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k {time.time()-t0:.0f}s", flush=True)
    return mae, rmse, r2

mae, rmse, r2 = train()
pd.DataFrame([{"config": "dual_qm_champion275", "n_feat": NQM, "n": N, "MAE": mae, "RMSE": rmse, "R2": r2}]).to_csv(
    f"{OUT}/gnn_dual_qm_champion275_results.csv", index=False)
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow_benchmark.db"); mlflow.set_experiment("exp1_gnn_dual_champion275")
with mlflow.start_run(run_name="GNN_dual_qm_champion275"):
    mlflow.log_params({"model": "dual_qm_champion275", "nfeat": NQM, "n": N, "arch": "dual_encoder_GINE"})
    mlflow.log_metrics({"test_mae": mae, "test_r2": r2})
print("ref: tabular champion MORDREDSLIM271_BDEGXTB test MAE 1.503 | old dual_qm(56feat) 1.646 "
      "| old tabular ensemble(72feat) 1.61", flush=True)
print("DONE", flush=True)
