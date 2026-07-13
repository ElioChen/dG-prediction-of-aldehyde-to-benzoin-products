#!/usr/bin/env python
"""Does adding REACTANT (aldehyde) descriptors improve the g-xTB->DFT correction?
ΔG_error = G_product_error - 2*G_aldehyde_error, so product-only features miss the
reactant-side contribution. Join aldehydes_all.csv (CHO-site descriptors) via donor_id
and compare product-only vs product+reactant feature sets (GBT & MLP, Δ-learning).
"""
import glob, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
R = "/scratch-shared/schen3/benzoin-dg"
H = f"{R}/data/cross_benzoin/homo_v6"

PROD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega",
  "xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH",
  "wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC",
  "fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L",
  "sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]

def split(n):
    i = np.random.default_rng(42).permutation(n); a,b=int(.7*n),int(.9*n); return i[:a],i[a:b],i[b:]

def run(d, feats, tag):
    tr,va,te = split(len(d))
    sc = StandardScaler().fit(d[feats].values[tr])
    Xtr,Xva,Xte = (sc.transform(d[feats].values[tr]), sc.transform(d[feats].values[va]), sc.transform(d[feats].values[te]))
    dtr,dva = d.delta.values[tr], d.delta.values[va]
    g,y = d.dG_gxtb_kcal.values[te], d.dG_orca_kcal.values[te]
    out={}
    gb=XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                    n_jobs=8,early_stopping_rounds=40,eval_metric="mae")
    gb.fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
    out["GBT"]=float(np.abs(g+gb.predict(Xte)-y).mean())
    ml=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=120,early_stopping=True,n_iter_no_change=8)
    ml.fit(Xtr,dtr); out["MLP"]=float(np.abs(g+ml.predict(Xte)-y).mean())
    print(f"  [{tag}] nfeat={len(feats)}  GBT={out['GBT']:.3f}  MLP={out['MLP']:.3f}",flush=True)
    return out

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).rename(columns={"id":"ald_id"})
    a=a.rename(columns={c:f"ald_{c}" for c in ALD}).drop_duplicates("ald_id")
    ALDp=[f"ald_{c}" for c in ALD]
    df=p.merge(dft,on="id")
    df["ald_id"]=df["donor_id"].astype("Int64")
    df=df.merge(a,on="ald_id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+PROD_QM+ALDp)
    df=df[df["dG_orca_kcal"].abs()<60].reset_index(drop=True)
    df["delta"]=df["dG_orca_kcal"]-df["dG_gxtb_kcal"]
    print("rows with product+reactant features:",len(df),flush=True)
    print("=== product-only vs product+reactant (test MAE, kcal/mol) ===",flush=True)
    po=run(df,PROD_QM,"product-only")
    pr=run(df,PROD_QM+ALDp,"product+reactant")
    print(f"\nΔMAE  GBT {po['GBT']:.3f}->{pr['GBT']:.3f} ({pr['GBT']-po['GBT']:+.3f})  "
          f"MLP {po['MLP']:.3f}->{pr['MLP']:.3f} ({pr['MLP']-po['MLP']:+.3f})",flush=True)
    # reactant-only sanity
    ro=run(df,ALDp,"reactant-only")
    pd.DataFrame([dict(featset="product_only",**po),dict(featset="product+reactant",**pr),
                  dict(featset="reactant_only",**ro)]).to_csv(f"{H}/viz_gxtb_20260625/ablation_reactant.csv",index=False)
    print("DONE",flush=True)

if __name__=="__main__":
    main()
