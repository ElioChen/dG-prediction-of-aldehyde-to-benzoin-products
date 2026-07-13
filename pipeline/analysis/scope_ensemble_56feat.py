#!/usr/bin/env python
"""56-feature (product+reactant) MLP / XGB / ENSEMBLE per scope {all, aromatic, aliphatic}.
Reports MAE + R². Answers: does the ensemble win on aromatic too, and how low does it go?"""
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

def fit_scope(d, name):
    d=d.reset_index(drop=True)
    i=np.random.default_rng(42).permutation(len(d)); ntr,nva=int(.7*len(d)),int(.9*len(d))
    tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    sc=StandardScaler().fit(d[FEATS].values[tr])
    Xtr,Xva,Xte=sc.transform(d[FEATS].values[tr]),sc.transform(d[FEATS].values[va]),sc.transform(d[FEATS].values[te])
    dtr=d.delta.values[tr]; dva=d.delta.values[va]; g=d.dG_gxtb_kcal.values[te]; y=d.dG_orca_kcal.values[te]
    ml=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10).fit(Xtr,dtr)
    gb=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,
                    min_child_weight=5,n_jobs=16,early_stopping_rounds=60,eval_metric="mae").fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
    pm,pg=ml.predict(Xte),gb.predict(Xte); pe=0.5*pm+0.5*pg
    def mr(p): e=g+p-y; return float(np.abs(e).mean()), float(1-(e**2).sum()/((y-y.mean())**2).sum())
    out={}
    for tag,p in [("MLP",pm),("XGB",pg),("ENSEMBLE",pe)]:
        mae,r2=mr(p); out[tag]=(mae,r2); print(f"  {name:9s} {tag:9s} MAE={mae:.3f} R2={r2:.3f}",flush=True)
    return out

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    print(f"labeled {len(df)}  by class {df.cls.value_counts().to_dict()}",flush=True)
    rows=[]
    for s,sub in [("all",df),("aromatic",df[df.cls=='aromatic']),("aliphatic",df[df.cls=='aliphatic'])]:
        o=fit_scope(sub,s)
        rows.append(dict(scope=s,n=len(sub),**{f"{k}_MAE":v[0] for k,v in o.items()},**{f"{k}_R2":v[1] for k,v in o.items()}))
    pd.DataFrame(rows).to_csv(f"{H}/viz_gxtb_20260625/scope_ensemble_56feat.csv",index=False)
    print("DONE",flush=True)

if __name__=="__main__": main()
