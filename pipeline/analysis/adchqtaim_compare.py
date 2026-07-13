#!/usr/bin/env python
"""Validation: does adding ADCH/QTAIM (product 16 + reactant 7) help vs the 56-QM baseline?
On the 2.5k stratified subset (P/B/S-enriched). 5-fold CV (small n). XGB handles NaN natively
(product ADCH only ~53% filled). Report MAE overall + P/B/S-subset + aliphatic.
"""
import glob, warnings, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"; D=f"{H}/viz_gxtb_20260625"
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new",
 "wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC",
 "dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int",
 "pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
 "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; FEATS=PROD_QM+ALDp

def main():
    aq=pd.read_parquet(f"{D}/adchqtaim_values.parquet")
    AQ=[c for c in aq.columns if c.startswith(("adch","qtaim","ald_adch","ald_qtaim"))]
    print("ADCH/QTAIM feature cols:",len(AQ),flush=True)
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))+sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/retry7200/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id",keep="last")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(aq,on="id",how="inner")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    df["pbs"]=df.smiles.str.contains(r"P|S|B|\[Si",regex=True)
    print("subset n",len(df),"P/B/S",int(df.pbs.sum()),flush=True)
    def cv(cols):
        kf=KFold(5,shuffle=True,random_state=42); e_all=[]; e_pbs=[]
        for tr,te in kf.split(df):
            m=XGBRegressor(n_estimators=600,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,
                n_jobs=24,eval_metric="mae").fit(df[cols].values[tr],df.delta.values[tr])
            yh=df.dG_gxtb_kcal.values[te]+m.predict(df[cols].values[te]); er=np.abs(yh-df.dG_orca_kcal.values[te])
            e_all.append(er); e_pbs.append(er[df.pbs.values[te]])
        e_all=np.concatenate(e_all); e_pbs=np.concatenate(e_pbs)
        return e_all.mean(),e_pbs.mean()
    a56=cv(FEATS); a79=cv(FEATS+AQ)
    print(f"\n=== 5-fold CV MAE (subset, XGB) ===",flush=True)
    print(f"  56 QM            all={a56[0]:.3f}  P/B/S={a56[1]:.3f}",flush=True)
    print(f"  56 + ADCH/QTAIM  all={a79[0]:.3f}  P/B/S={a79[1]:.3f}",flush=True)
    print(f"  Δ                all={a79[0]-a56[0]:+.3f}  P/B/S={a79[1]-a56[1]:+.3f}",flush=True)
    pd.DataFrame([{"featset":"56","all":a56[0],"pbs":a56[1]},{"featset":"56+ADCHQTAIM","all":a79[0],"pbs":a79[1]}]).to_csv(f"{D}/adchqtaim_compare.csv",index=False)
    print("DONE",flush=True)

if __name__=="__main__": main()
