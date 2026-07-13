#!/usr/bin/env python
"""Are the 56 features enough / redundant / room for more? Evidence:
  (1) XGB gain importance — which features carry signal, is there dead weight.
  (2) top-k ablation — does MAE plateau before 56 (redundant) or keep dropping (more would help)?
  (3) residual vs scope/size — where the unexplained error concentrates.
"""
import glob, warnings, numpy as np, pandas as pd
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

def xgb(): return XGBRegressor(n_estimators=800,max_depth=6,learning_rate=0.03,subsample=0.8,
    colsample_bytree=0.8,n_jobs=16,early_stopping_rounds=50,eval_metric="mae")

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    print("n",len(df),flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df))
    tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    sc=StandardScaler().fit(df[FEATS].values[tr])
    def mk(cols):
        idx=[FEATS.index(c) for c in cols]
        return sc.transform(df[FEATS].values[tr])[:,idx],sc.transform(df[FEATS].values[va])[:,idx],sc.transform(df[FEATS].values[te])[:,idx]
    g=df.dG_gxtb_kcal.values[te]; y=df.dG_orca_kcal.values[te]; dtr=df.delta.values[tr]; dva=df.delta.values[va]
    # full 56 importance
    Xtr,Xva,Xte=mk(FEATS); m=xgb().fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
    imp=pd.Series(m.feature_importances_,index=FEATS).sort_values(ascending=False)
    full_mae=float(np.abs(g+m.predict(Xte)-y).mean())
    print(f"\nFULL 56  MAE={full_mae:.3f}",flush=True)
    print("=== top-15 importances ===\n"+imp.head(15).to_string(),flush=True)
    print(f"\nbottom-10 importances sum = {imp.tail(10).sum():.3f} (of 1.0)",flush=True)
    # top-k ablation
    print("\n=== top-k ablation (does MAE plateau?) ===",flush=True)
    for k in [5,10,15,20,30,40,56]:
        cols=list(imp.head(k).index); Xtr,Xva,Xte=mk(cols)
        mk_=xgb().fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
        mae=float(np.abs(g+mk_.predict(Xte)-y).mean()); print(f"  top-{k:2d}  MAE={mae:.3f}",flush=True)
    # residual by scope + size tertile
    Xtr,Xva,Xte=mk(FEATS); m=xgb().fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False); res=np.abs(g+m.predict(Xte)-y)
    cl=df.cls.values[te]; sasa=df.SASA_total.values[te]
    print("\n=== residual |err| by scope ===",flush=True)
    for s in ["aromatic","aliphatic"]:
        mk2=cl==s; print(f"  {s:9s} mean|err|={res[mk2].mean():.3f}  p90={np.percentile(res[mk2],90):.2f}",flush=True)
    q=np.quantile(sasa,[.33,.66]);
    for lab,mask in [("small",sasa<q[0]),("mid",(sasa>=q[0])&(sasa<q[1])),("large",sasa>=q[1])]:
        print(f"  size-{lab:5s} mean|err|={res[mask].mean():.3f}",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
