#!/usr/bin/env python
"""Headroom probe for the 56-feature (product+reactant) Δ-correction.
Reports MAE AND R² for the current champion and several stronger tabular models
(deeper MLP, larger/deeper XGB, MLP+XGB ensemble) to see if 1.73 has room to drop.
"""
import glob, warnings, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new",
 "wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC",
 "dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int",
 "pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
 "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; FEATS=PROD_QM+ALDp

def metrics(g,m,X,y):
    yh=g+m.predict(X); e=yh-y
    return float(np.abs(e).mean()), float(1-(e**2).sum()/((y-y.mean())**2).sum()), yh

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    print("n labeled:",len(df),flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.2*len(df))
    tr,va,te=i[:ntr],i[ntr:ntr+nva],i[ntr+nva:]
    sc=StandardScaler().fit(df[FEATS].values[tr])
    Xtr,Xva,Xte=sc.transform(df[FEATS].values[tr]),sc.transform(df[FEATS].values[va]),sc.transform(df[FEATS].values[te])
    dtr,dva=df.delta.values[tr],df.delta.values[va]; gte,yte=df.dG_gxtb_kcal.values[te],df.dG_orca_kcal.values[te]
    res=[]; preds={}
    def add(name,m):
        mae,r2,yh=metrics(gte,m,Xte,yte); res.append((name,mae,r2)); preds[name]=yh
        print(f"  {name:26s} MAE={mae:.3f}  R2={r2:.3f}",flush=True)
    m=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10).fit(Xtr,dtr); add("MLP(256,128) [champion]",m)
    m=MLPRegressor(hidden_layer_sizes=(512,256,128),alpha=1e-4,max_iter=300,early_stopping=True,n_iter_no_change=12).fit(Xtr,dtr); add("MLP(512,256,128) deeper",m)
    m=XGBRegressor(n_estimators=1200,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,n_jobs=16,early_stopping_rounds=60,eval_metric="mae").fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False); add("XGB(1200,d6,lr.03)",m)
    m=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,n_jobs=16,early_stopping_rounds=60,eval_metric="mae").fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False); add("XGB(1500,d8,lr.02)",m)
    # ensemble best MLP + best XGB
    ens=0.5*preds["MLP(512,256,128) deeper"]+0.5*preds["XGB(1200,d6,lr.03)"]
    e=ens-yte; print(f"  {'ENSEMBLE MLP+XGB':26s} MAE={np.abs(e).mean():.3f}  R2={1-(e**2).sum()/((yte-yte.mean())**2).sum():.3f}",flush=True)
    pd.DataFrame(res,columns=["model","MAE","R2"]).to_csv(f"{H}/viz_gxtb_20260625/headroom_probe.csv",index=False)
    print("DONE",flush=True)

if __name__=="__main__": main()
