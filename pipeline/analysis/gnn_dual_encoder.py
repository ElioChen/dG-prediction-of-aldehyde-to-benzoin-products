#!/usr/bin/env python
"""Dual-encoder Δ-GNN: encode PRODUCT graph and ALDEHYDE graph separately, combine
mirroring the thermodynamic cycle  ΔG = G_prod − 2·G_ald.

Target Δ = DFT − g-xTB.  Readout = [h_prod, h_ald, h_prod − 2·h_ald] (+QM optional).
Configs (same random 70/20/10 seed 42, n≈DFT-labelled products):
  product_only : single GINE on product graph (reference; matches arch-study gine ~2.66)
  dual         : product + aldehyde graphs, cycle-combine
  dual_qm      : dual + 34 product QM descriptors at readout
Tests the question: should the GNN use product / reactant / both SMILES?
GPU node, envs/gnn (torch + PyG2.8 + rdkit).
"""
import glob, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
RDLogger.DisableLog("rdApp.*")

ROOT = "/scratch-shared/schen3/benzoin-dg"
PROD = f"{ROOT}/data/cross_benzoin/homo_v6/products_all.csv"
DFTDIR = f"{ROOT}/data/raw/dft_sp_funnelv3"
OUT = f"{ROOT}/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
CACHE = f"{OUT}/gnn_dual_cache_56full.pt"  # full 100% labels
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket",
  "wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC",
  "fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5",
  "SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
# + 22 reactant (aldehyde) QM, joined via donor_id -> full 56-feature readout (info parity with tabular)
ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
  "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp = [f"ald_{c}" for c in ALD]
FEATS = QM + ALDp
NQM = len(FEATS)
ALDCSV = f"{ROOT}/data/cross_benzoin/homo_v6/aldehydes_all.csv"
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

if os.path.exists(CACHE):
    blob=torch.load(CACHE, weights_only=False); pairs=blob["pairs"]; qm_mean=blob["qm_mean"]; qm_std=blob["qm_std"]
    print("loaded cache", len(pairs), flush=True)
else:
    dft=pd.read_parquet(f"{DFTDIR}/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")  # consolidated 100% labels
    p=pd.read_csv(PROD,usecols=["id","donor_id","smiles","donor_smiles","dG_gxtb_kcal"]+QM,low_memory=False)
    a=pd.read_csv(ALDCSV,usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    df=p.merge(dft,on="id"); df["ald_id"]=df["donor_id"].astype("Int64"); df=df.merge(a,on="ald_id",how="left")
    df=df.dropna(subset=["smiles","donor_smiles","dG_gxtb_kcal","dG_orca_kcal"]+FEATS)
    df=df[df["dG_orca_kcal"].abs()<60].reset_index(drop=True)
    qm_mean=df[FEATS].mean().values; qm_std=df[FEATS].std().replace(0,1).values
    qmz=((df[FEATS].values-qm_mean)/qm_std)
    print("building", len(df), "pairs", flush=True); t0=time.time(); pairs=[]
    for k in range(len(df)):
        gp=graph(df.smiles.iloc[k]); ga=graph(df.donor_smiles.iloc[k])
        if gp is None or ga is None: continue
        d=PairData(); d.x_p,d.edge_index_p,d.edge_attr_p=gp; d.x_a,d.edge_index_a,d.edge_attr_a=ga
        d.y=torch.tensor([float(df.dG_orca_kcal.iloc[k])-float(df.dG_gxtb_kcal.iloc[k])],dtype=torch.float)
        d.g0=torch.tensor([float(df.dG_gxtb_kcal.iloc[k])],dtype=torch.float)
        d.qm=torch.tensor(qmz[k],dtype=torch.float).view(1,-1)
        pairs.append(d)
        if k%20000==0: print(f"  {k} ({time.time()-t0:.0f}s)",flush=True)
    torch.save({"pairs":pairs,"qm_mean":qm_mean,"qm_std":qm_std},CACHE)
    print("built+cached", len(pairs), flush=True)

N=len(pairs)
rng=np.random.default_rng(42); idx=rng.permutation(N)
ntr,nva=int(.7*N),int(.2*N); tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
print(f"split {len(tr)}/{len(va)}/{len(te)}",flush=True)
ys=torch.tensor([pairs[i].y.item() for i in tr]); ym,ysd=ys.mean().item(),ys.std().item()
for d in pairs: d.ystd=(d.y-ym)/ysd
def loader(ix,sh): return DataLoader([pairs[i] for i in ix],batch_size=256,shuffle=sh,follow_batch=["x_p","x_a"])
AD=pairs[0].x_p.shape[1]; BD=pairs[0].edge_attr_p.shape[1]

class Enc(nn.Module):
    def __init__(self,h=128,L=4):
        super().__init__(); self.ne=nn.Linear(AD,h); self.ee=nn.Linear(BD,h)
        self.cv=nn.ModuleList(); self.bn=nn.ModuleList()
        for _ in range(L):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h)),edge_dim=h)); self.bn.append(nn.BatchNorm1d(h))
    def forward(self,x,ei,ea,batch):
        x=self.ne(x); e=self.ee(ea)
        for c,b in zip(self.cv,self.bn): x=x+F.relu(b(c(x,ei,e)))
        return torch.cat([global_mean_pool(x,batch),global_add_pool(x,batch)],1)

class DualGNN(nn.Module):
    def __init__(self,kind,h=128,L=4):
        super().__init__(); self.kind=kind
        self.encP=Enc(h,L)
        self.encA=Enc(h,L) if kind!="product_only" else None
        din = 2*h if kind=="product_only" else 6*h
        if kind=="dual_qm": din+=NQM
        self.head=nn.Sequential(nn.Linear(din,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,1))
    def forward(self,b):
        hP=self.encP(b.x_p,b.edge_index_p,b.edge_attr_p,b.x_p_batch)
        if self.kind=="product_only":
            h=hP
        else:
            hA=self.encA(b.x_a,b.edge_index_a,b.edge_attr_a,b.x_a_batch)
            h=torch.cat([hP,hA,hP-2*hA],1)          # thermodynamic-cycle combine
            if self.kind=="dual_qm": h=torch.cat([h,b.qm.view(-1,NQM)],1)
        return self.head(h).squeeze(-1)

def evl(model,ld):
    model.eval(); ps,ts,gs=[],[],[]
    for b in ld:
        b=b.to(dev)
        with torch.no_grad(): o=model(b)
        ps.append(o.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu())
    pdl=torch.cat(ps).numpy()*ysd+ym; td=torch.cat(ts).numpy()*ysd+ym; g0=torch.cat(gs).numpy()
    yh,yt=g0+pdl,g0+td; r2=1-((yt-yh)**2).sum()/((yt-yt.mean())**2).sum()
    return float(np.abs(yt-yh).mean()),float(np.sqrt(((yt-yh)**2).mean())),r2

def train(kind):
    torch.manual_seed(0); model=DualGNN(kind).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=4)
    tl,vl=loader(tr,True),loader(va,False); best,bs,pat=1e9,None,0; t0=time.time()
    for ep in range(120):
        model.train()
        for b in tl:
            b=b.to(dev); opt.zero_grad(); loss=F.mse_loss(model(b),b.ystd); loss.backward(); opt.step()
        vm=evl(model,vl)[0]; sch.step(vm)
        if vm<best-1e-4: best,bs,pat=vm,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else: pat+=1
        if ep%10==0 or pat==0: print(f"  [{kind}] ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t0:.0f}s",flush=True)
        if pat>=15: break
    model.load_state_dict(bs); mae,rmse,r2=evl(model,loader(te,False))
    print(f"==> {kind:13s} TEST MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}  params={sum(p.numel() for p in model.parameters())/1e3:.0f}k  {time.time()-t0:.0f}s",flush=True)
    return dict(config=kind,MAE=mae,RMSE=rmse,R2=r2)

rows=[train(k) for k in ["product_only","dual","dual_qm"]]
res=pd.DataFrame(rows).sort_values("MAE")
res.to_csv(f"{OUT}/gnn_dual_results_56full.csv",index=False)
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow_benchmark.db"); mlflow.set_experiment("exp1_gnn_dual_full")
for r in rows:
    with mlflow.start_run(run_name=f"GNN_{r['config']}_56full"):
        mlflow.log_params({"model":r["config"],"nfeat":56,"n":N,"arch":"dual_encoder_GINE"}); mlflow.log_metrics({"test_mae":r["MAE"],"test_r2":r["R2"]})
print("\n=== DUAL-ENCODER (product vs reactant vs both) test MAE ===\n"+res.to_string(index=False),flush=True)
print("ref: single product GINE 2.66 | product+QM hybrid 2.13 | GBT 2.46",flush=True)
print("DONE",flush=True)
