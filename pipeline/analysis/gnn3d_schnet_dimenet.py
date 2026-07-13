#!/usr/bin/env python
"""3D GNNs on the PRODUCT geometry: can reading xyz directly beat the 56-feature tabular (1.73)?
Δ-learning (DFT − g-xTB). Architectures: SchNet, DimeNet++, and an equivariant model
(PaiNN if importable, else ViSNet). Same 60k subset + 70/20/10 seed 42; an MLP on the same
subset's 56 QM features is included as the fair tabular baseline. Reports MAE + R².
GPU node, envs/gnn.
"""
import glob, os, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
warnings.filterwarnings("ignore")

# Use native torch_cluster.radius_graph if present (nequip env); else pure-torch fallback (gnn env).
try:
    import torch_cluster  # noqa: F401
    print("torch_cluster present -> native radius_graph", flush=True)
except Exception:
    def radius_graph_cdist(x, r, batch=None, loop=False, max_num_neighbors=64,
                           flow="source_to_target", **kw):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        rows, cols = [], []
        for b in torch.unique(batch):
            m = (batch == b).nonzero(as_tuple=True)[0]
            d = torch.cdist(x[m], x[m])
            adj = d <= r
            if not loop:
                adj.fill_diagonal_(False)
            s, t = adj.nonzero(as_tuple=True)
            rows.append(m[s]); cols.append(m[t])
        row = torch.cat(rows); col = torch.cat(cols)
        return torch.stack([row, col], 0) if flow == "source_to_target" else torch.stack([col, row], 0)
    import torch_geometric.nn.models.schnet as _schnet
    import torch_geometric.nn.models.dimenet as _dimenet
    _schnet.radius_graph = radius_graph_cdist
    _dimenet.radius_graph = radius_graph_cdist
    print("torch_cluster MISSING -> pure-torch radius_graph patch", flush=True)

ROOT="/scratch-shared/schen3/benzoin-dg"; H=f"{ROOT}/data/cross_benzoin/homo_v6"
OUT=f"{H}/viz_gxtb_20260625"; CACHE=f"{OUT}/gnn3d_cache_60k.pt"
N_SUB=60000
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device",dev,flush=True)
SYM2Z={'H':1,'B':5,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Se':34,'Br':35,'I':53}

PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new",
 "wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC",
 "dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int",
 "pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
 "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; FEATS=PROD_QM+ALDp

def read_xyz(path):
    try:
        with open(path) as f: ls=f.read().split("\n")
        n=int(ls[0]); z=[]; pos=[]
        for i in range(2,2+n):
            p=ls[i].split()
            if len(p)<4: return None
            z.append(SYM2Z.get(p[0],0)); pos.append([float(p[1]),float(p[2]),float(p[3])])
        if 0 in z: return None
        return torch.tensor(z,dtype=torch.long), torch.tensor(pos,dtype=torch.float)
    except Exception: return None

if os.path.exists(CACHE):
    blob=torch.load(CACHE,weights_only=False); data=blob["data"]; print("loaded cache",len(data),flush=True)
else:
    fs=sorted(glob.glob(f"{ROOT}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    cols=["id","donor_id","xyz_file","dG_gxtb_kcal"]+PROD_QM
    p=pd.read_csv(f"{H}/products_all.csv",usecols=cols,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal","xyz_file"]+FEATS); df=df[df.dG_orca_kcal.abs()<60]
    df=df.sample(n=min(N_SUB,len(df)),random_state=42).reset_index(drop=True)
    qm_mean=df[FEATS].mean().values; qm_std=df[FEATS].std().replace(0,1).values
    print("reading xyz for",len(df),"mols",flush=True); t0=time.time(); data=[]
    for k in range(len(df)):
        g=read_xyz(df.xyz_file.iloc[k])
        if g is None: continue
        z,pos=g
        d=Data(z=z,pos=pos,
               y=torch.tensor([float(df.dG_orca_kcal.iloc[k])-float(df.dG_gxtb_kcal.iloc[k])],dtype=torch.float),
               g0=torch.tensor([float(df.dG_gxtb_kcal.iloc[k])],dtype=torch.float),
               qm=torch.tensor((df[FEATS].iloc[k].values-qm_mean)/qm_std,dtype=torch.float).view(1,-1))
        data.append(d)
        if k%10000==0: print(f"  {k} ({time.time()-t0:.0f}s)",flush=True)
    torch.save({"data":data},CACHE); print("built+cached",len(data),flush=True)

N=len(data); rng=np.random.default_rng(42); idx=rng.permutation(N)
ntr,nva=int(.7*N),int(.2*N); tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
ys=torch.tensor([data[i].y.item() for i in tr]); ym,ysd=ys.mean().item(),ys.std().item()
for d in data: d.ystd=(d.y-ym)/ysd
def loader(ix,sh,bs=128): return DataLoader([data[i] for i in ix],batch_size=bs,shuffle=sh)
print(f"split {len(tr)}/{len(va)}/{len(te)}  (subset N={N})",flush=True)

NQM=len(FEATS)
def make(name):
    from torch_geometric.nn.models import SchNet, DimeNetPlusPlus
    if name=="SchNet": return SchNet(hidden_channels=128,num_filters=128,num_interactions=6,num_gaussians=50,cutoff=10.0)
    if name=="DimeNet++": return DimeNetPlusPlus(hidden_channels=128,out_channels=1,num_blocks=4,int_emb_size=64,
                          basis_emb_size=8,out_emb_channels=256,num_spherical=7,num_radial=6,cutoff=5.0)
    if name=="equivariant":
        try:
            from torch_geometric.nn.models import ViSNet
            return ("ViSNet", ViSNet(hidden_channels=128,num_layers=6,num_heads=8,cutoff=5.0))
        except Exception as e:
            print("ViSNet unavailable:",e,flush=True); return None
    return None

def fwd(model,b):
    out=model(b.z,b.pos,b.batch)
    if isinstance(out,tuple): out=out[0]
    return out.view(-1)

def run3d(name):
    obj=make(name)
    if obj is None: return None
    label=name
    if isinstance(obj,tuple): label,model=obj
    else: model=obj
    model=model.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=4)
    tl,vl=loader(tr,True),loader(va,False); best,bs,pat=1e9,None,0; t0=time.time()
    def evl(ld):
        model.eval(); ps,ts,gs=[],[],[]
        for b in ld:
            b=b.to(dev)
            with torch.no_grad(): o=fwd(model,b)
            ps.append(o.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu())
        pdl=torch.cat(ps).numpy()*ysd+ym; td=torch.cat(ts).numpy()*ysd+ym; g0=torch.cat(gs).numpy()
        yh,yt=g0+pdl,g0+td; return float(np.abs(yt-yh).mean()),float(1-((yt-yh)**2).sum()/((yt-yt.mean())**2).sum())
    for ep in range(80):
        model.train()
        for b in tl:
            b=b.to(dev); opt.zero_grad(); loss=F.mse_loss(fwd(model,b),b.ystd); loss.backward(); opt.step()
        vm,_=evl(vl); sch.step(vm)
        if vm<best-1e-4: best,pat=vm,0
        else: pat+=1
        if ep%5==0 or pat==0: print(f"  [{label}] ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t0:.0f}s",flush=True)
        if pat>=12: break
    mae,r2=evl(loader(te,False))
    print(f"==> {label:12s} TEST MAE={mae:.3f} R2={r2:.3f}  {time.time()-t0:.0f}s",flush=True)
    return dict(model=label,MAE=mae,R2=r2)

# tabular baseline on the SAME subset (fair comparison)
def tab_baseline():
    from sklearn.neural_network import MLPRegressor
    Xtr=np.stack([data[i].qm.numpy().ravel() for i in tr]); Xte=np.stack([data[i].qm.numpy().ravel() for i in te])
    dtr=np.array([data[i].y.item() for i in tr]); gte=np.array([data[i].g0.item() for i in te])
    yte=np.array([data[i].g0.item()+data[i].y.item() for i in te])
    m=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10).fit(Xtr,dtr)
    yh=gte+m.predict(Xte); mae=float(np.abs(yh-yte).mean()); r2=float(1-((yh-yte)**2).sum()/((yte-yte.mean())**2).sum())
    print(f"==> {'MLP(56feat)':12s} TEST MAE={mae:.3f} R2={r2:.3f}  (same 60k subset)",flush=True)
    return dict(model="MLP_56feat_same60k",MAE=mae,R2=r2)

rows=[]
try: rows.append(tab_baseline())
except Exception as e: print(f"tab_baseline skipped (env lacks sklearn?): {e}  [ref 60k MLP=1.83]",flush=True)
for nm in ["SchNet","DimeNet++","equivariant"]:
    try: r=run3d(nm); rows.append(r) if r else None
    except Exception as e: print(f"!! {nm} FAILED: {e}",flush=True)
res=pd.DataFrame([r for r in rows if r]).sort_values("MAE")
res.to_csv(f"{OUT}/gnn3d_results{os.environ.get('GNN3D_TAG','')}.csv",index=False)
print("\n=== 3D-GNN vs tabular (same 60k, test) ===\n"+res.to_string(index=False),flush=True)
print("ref(full 150k): MLP 56feat 1.73 | dual_qm 2.02 | gine_hybrid 2.13",flush=True)
print("DONE",flush=True)
