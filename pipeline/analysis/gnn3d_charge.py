#!/usr/bin/env python
"""Does adding per-atom partial charge as a node feature help the 3D GNN?
Custom SchNet (continuous-filter conv) with optional charge injection into the atom
embedding. Charges = RDKit Gasteiger, derived from cached (z,pos) via xyz2mol (no xyz re-read).
Ablation: SchNet vs SchNet+charge on the same 60k subset. Δ-learning (DFT - g-xTB).
Run in gnn env (rdkit present; torch_cluster absent -> pure-torch radius_graph). GPU.
"""
import os, time, warnings, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
warnings.filterwarnings("ignore")
H="/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6"; OUT=f"{H}/viz_gxtb_20260625"
CACHE=f"{OUT}/gnn3d_cache_60k.pt"; QCACHE=f"{OUT}/gnn3d_cache_60k_q.pt"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device",dev,flush=True)

def radius_graph(x,r,batch,loop=False):
    rows,cols=[],[]
    for b in torch.unique(batch):
        m=(batch==b).nonzero(as_tuple=True)[0]; d=torch.cdist(x[m],x[m]); adj=d<=r
        if not loop: adj.fill_diagonal_(False)
        s,t=adj.nonzero(as_tuple=True); rows.append(m[s]); cols.append(m[t])
    return torch.stack([torch.cat(rows),torch.cat(cols)],0)

# ---- build charge-augmented cache ----
if os.path.exists(QCACHE):
    data=torch.load(QCACHE,weights_only=False)["data"]; print("loaded q-cache",len(data),flush=True)
else:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds, rdPartialCharges
    RDLogger.DisableLog('rdApp.*')
    Z2SYM={1:'H',5:'B',6:'C',7:'N',8:'O',9:'F',14:'Si',15:'P',16:'S',17:'Cl',34:'Se',35:'Br',53:'I'}
    base=torch.load(CACHE,weights_only=False)["data"]; print("loaded base",len(base),flush=True)
    def gasteiger(z,pos):
        n=len(z); lines=[str(n),""]
        for zi,p in zip(z.tolist(),pos.tolist()):
            lines.append(f"{Z2SYM.get(zi,'C')} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
        try:
            m=Chem.MolFromXYZBlock("\n".join(lines))
            rdDetermineBonds.DetermineBonds(m,charge=0)
            rdPartialCharges.ComputeGasteigerCharges(m)
            q=[a.GetDoubleProp('_GasteigerCharge') for a in m.GetAtoms()]
            q=np.nan_to_num(np.array(q,dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
            if len(q)!=n: return None
            return q
        except Exception: return None
    t0=time.time(); data=[]; ok=0
    for k,d in enumerate(base):
        q=gasteiger(d.z,d.pos)
        d.q=torch.tensor(q if q is not None else np.zeros(len(d.z),dtype=np.float32))
        d.has_q=q is not None; ok+=int(q is not None); data.append(d)
        if k%10000==0: print(f"  {k} ({time.time()-t0:.0f}s) q-ok={ok}",flush=True)
    torch.save({"data":data},QCACHE); print(f"built q-cache {len(data)} (charge ok {ok})",flush=True)

N=len(data); rng=np.random.default_rng(42); idx=rng.permutation(N)
ntr,nva=int(.7*N),int(.2*N); tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
ys=torch.tensor([data[i].y.item() for i in tr]); ym,ysd=ys.mean().item(),ys.std().item()
for d in data: d.ystd=(d.y-ym)/ysd
def loader(ix,sh): return DataLoader([data[i] for i in ix],batch_size=128,shuffle=sh)
print(f"split {len(tr)}/{len(va)}/{len(te)}",flush=True)

class RBF(nn.Module):
    def __init__(s,K=50,cut=10.0):
        super().__init__(); s.register_buffer("c",torch.linspace(0,cut,K)); s.g=-0.5/((cut/K)**2)
    def forward(s,d): return torch.exp(s.g*(d.unsqueeze(-1)-s.c)**2)
class CFConv(nn.Module):
    def __init__(s,h,K):
        super().__init__(); s.l1=nn.Linear(h,h,bias=False); s.f=nn.Sequential(nn.Linear(K,h),nn.SiLU(),nn.Linear(h,h)); s.l2=nn.Linear(h,h)
    def forward(s,x,ei,rbf):
        W=s.f(rbf); xj=s.l1(x)[ei[1]]*W
        agg=torch.zeros_like(x).index_add_(0,ei[0],xj); return s.l2(agg)
class SchNetQ(nn.Module):
    def __init__(s,h=128,L=6,K=50,cut=10.0,use_q=False):
        super().__init__(); s.use_q=use_q; s.cut=cut; s.emb=nn.Embedding(100,h)
        if use_q: s.ql=nn.Linear(1,h)
        s.rbf=RBF(K,cut); s.cv=nn.ModuleList([CFConv(h,K) for _ in range(L)])
        s.out=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(s,b):
        h=s.emb(b.z)
        if s.use_q: h=h+s.ql(b.q.view(-1,1))
        ei=radius_graph(b.pos,s.cut,b.batch); d=(b.pos[ei[0]]-b.pos[ei[1]]).norm(dim=1); rbf=s.rbf(d)
        for c in s.cv: h=h+c(h,ei,rbf)
        return global_add_pool(s.out(h),b.batch).squeeze(-1)

def evl(model,ld):
    model.eval(); ps,ts,gs=[],[],[]
    for b in ld:
        b=b.to(dev)
        with torch.no_grad(): o=model(b)
        ps.append(o.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu())
    p=torch.cat(ps).numpy()*ysd+ym; t=torch.cat(ts).numpy()*ysd+ym; g=torch.cat(gs).numpy()
    yh,yt=g+p,g+t; return float(np.abs(yt-yh).mean()),float(1-((yt-yh)**2).sum()/((yt-yt.mean())**2).sum())

def train(use_q):
    name="SchNet+charge" if use_q else "SchNet(z+pos)"
    torch.manual_seed(0); model=SchNetQ(use_q=use_q).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=4)
    tl,vl=loader(tr,True),loader(va,False); best,bs,pat=1e9,None,0; t0=time.time()
    for ep in range(80):
        model.train()
        for b in tl:
            b=b.to(dev); opt.zero_grad(); loss=F.mse_loss(model(b),b.ystd); loss.backward(); opt.step()
        vm,_=evl(model,vl); sch.step(vm)
        if vm<best-1e-4: best,bs,pat=vm,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else: pat+=1
        if ep%5==0 or pat==0: print(f"  [{name}] ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t0:.0f}s",flush=True)
        if pat>=12: break
    model.load_state_dict(bs); mae,r2=evl(model,loader(te,False))
    print(f"==> {name:16s} TEST MAE={mae:.3f} R2={r2:.3f}  {time.time()-t0:.0f}s",flush=True)
    return dict(model=name,MAE=mae,R2=r2)

import pandas as pd
rows=[train(False),train(True)]
pd.DataFrame(rows).to_csv(f"{OUT}/gnn3d_charge_results.csv",index=False)
print("\n=== SchNet ± charge (same 60k) ===\n"+pd.DataFrame(rows).to_string(index=False),flush=True)
print("ref: same-60k MLP(56feat)=1.83 | full MLP+XGB ens=1.61",flush=True)
print("DONE",flush=True)
