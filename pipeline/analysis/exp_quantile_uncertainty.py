#!/usr/bin/env python
"""[3] Better uncertainty: XGBoost quantile regression (q05/q50/q95) -> calibrated prediction
interval as the routing signal. Compare confident-MAE + PI coverage vs the ensemble-std baseline.
Logs to MLflow."""
import warnings, numpy as np, pandas as pd, mlflow
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
mlflow.set_tracking_uri(f"sqlite:///{R}/mlflow_benchmark.db"); mlflow.set_experiment("exp3_uncertainty")
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; GLOB=[f"g_{k}" for k in ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide","has_P","has_B","has_S","has_Si","has_halogen"]]
FEATS=PROD_QM+ALDp+GLOB

def main():
    dft=pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    g=pd.read_parquet(f"{H}/viz_gxtb_20260625/global_descriptors_full.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(g,on="smiles",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df)); tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    sc=StandardScaler().fit(df[FEATS].values[tr]); Xtr,Xva,Xte=sc.transform(df[FEATS].values[tr]),sc.transform(df[FEATS].values[va]),sc.transform(df[FEATS].values[te])
    dtr=df.delta.values[tr]; g0=df.dG_gxtb_kcal.values[te]; y=df.dG_orca_kcal.values[te]
    print("n",len(df),flush=True)
    # quantile regression
    def qreg(al): return XGBRegressor(objective="reg:quantileerror",quantile_alpha=al,n_estimators=800,max_depth=7,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,n_jobs=24).fit(Xtr,dtr)
    q05,q50,q95=[qreg(a).predict(Xte) for a in (0.05,0.5,0.95)]
    yh=g0+q50; err=np.abs(yh-y); width=q95-q05
    cover=float(((y>=g0+q05)&(y<=g0+q95)).mean())   # should ~0.90 if calibrated
    # routing by interval width (top 15% widest -> route)
    thrW=np.quantile(width,0.85); confW=width<thrW
    # baseline: ensemble-std (MLP + XGB)
    ml=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10).fit(Xtr,dtr)
    gb=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,n_jobs=24,eval_metric="mae").fit(Xtr,dtr)
    std=np.abs(ml.predict(Xte)-gb.predict(Xte)); thrS=np.quantile(std,0.85); confS=std<thrS
    err_ens=np.abs(g0+0.5*ml.predict(Xte)+0.5*gb.predict(Xte)-y)
    def sep(err,conf): return float(err[conf].mean()),float(err[~conf].mean())
    qc,qr=sep(err,confW); sc_,sr=sep(err_ens,confS)
    print(f"quantile PI coverage(90% target)={cover:.3f}",flush=True)
    print(f"  quantile-width routing: conf-MAE={qc:.3f} routed-MAE={qr:.3f} (sep {qr/qc:.2f}x)",flush=True)
    print(f"  ensemble-std routing:   conf-MAE={sc_:.3f} routed-MAE={sr:.3f} (sep {sr/sc_:.2f}x)",flush=True)
    with mlflow.start_run(run_name="quantile_interval"):
        mlflow.log_params({"method":"xgb_quantile","n":len(df)}); mlflow.log_metrics({"pi_coverage_90":cover,"conf_mae":qc,"routed_mae":qr,"separation":qr/qc})
    with mlflow.start_run(run_name="ensemble_std_baseline"):
        mlflow.log_params({"method":"mlp_xgb_std","n":len(df)}); mlflow.log_metrics({"conf_mae":sc_,"routed_mae":sr,"separation":sr/sc_})
    print("DONE",flush=True)

if __name__=="__main__": main()
