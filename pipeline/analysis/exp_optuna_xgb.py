#!/usr/bin/env python
"""Hyperparameter search (Optuna/TPE) for a REGULARIZED XGB that keeps low test MAE with a
smaller train-test gap (the d10 default overfits: train 0.56 vs test 1.59). Optimize VALIDATION
MAE (test held out). Search regularization knobs. Report best + train/val/test gap. Logs to MLflow."""
import warnings, numpy as np, pandas as pd, mlflow, optuna
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
mlflow.set_tracking_uri(f"sqlite:///{R}/mlflow_benchmark.db"); mlflow.set_experiment("exp6_optuna_xgb")
N_TRIALS=60
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; GLOB=[f"g_{k}" for k in ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide","has_P","has_B","has_S","has_Si","has_halogen"]]
F=PROD_QM+ALDp+GLOB

def main():
    dft=pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    g=pd.read_parquet(f"{H}/viz_gxtb_20260625/global_descriptors_full.parquet")
    df=p.merge(dft,on="id");df["ald_id"]=df.donor_id.astype("Int64");df=df.merge(a,on="ald_id",how="left").merge(g,on="smiles",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+F);df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True);df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    i=np.random.default_rng(42).permutation(len(df));ntr,nva=int(.7*len(df)),int(.9*len(df));tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    sc=StandardScaler().fit(df[F].values[tr]);Xtr,Xva,Xte=sc.transform(df[F].values[tr]),sc.transform(df[F].values[va]),sc.transform(df[F].values[te])
    dtr,dva=df.delta.values[tr],df.delta.values[va]
    G=df.dG_gxtb_kcal.values; Y=df.dG_orca_kcal.values
    def mae(ix,m,X): return float(np.abs(G[ix]+m.predict(X)-Y[ix]).mean())
    print(f"n {len(df)} split {len(tr)}/{len(va)}/{len(te)}",flush=True)
    def objective(t):
        params=dict(n_estimators=3000,
            max_depth=t.suggest_int("max_depth",4,9),
            learning_rate=t.suggest_float("learning_rate",0.01,0.1,log=True),
            min_child_weight=t.suggest_int("min_child_weight",1,30),
            subsample=t.suggest_float("subsample",0.5,0.95),
            colsample_bytree=t.suggest_float("colsample_bytree",0.4,0.95),
            reg_lambda=t.suggest_float("reg_lambda",0.1,50,log=True),
            reg_alpha=t.suggest_float("reg_alpha",1e-3,10,log=True),
            gamma=t.suggest_float("gamma",0,5),
            n_jobs=24,early_stopping_rounds=50,eval_metric="mae")
        m=XGBRegressor(**params).fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
        return mae(va,m,Xva)
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
    bp=study.best_params; print("best params",bp,flush=True)
    m=XGBRegressor(n_estimators=3000,n_jobs=24,early_stopping_rounds=50,eval_metric="mae",**bp).fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
    mtr,mva,mte=mae(tr,m,Xtr),mae(va,m,Xva),mae(te,m,Xte)
    gap=mte-mtr
    print(f"\nTUNED XGB: train {mtr:.3f} | val {mva:.3f} | test {mte:.3f}  (gap test-train={gap:.3f})",flush=True)
    print(f"vs default d10: train 0.56 test 1.59 (gap 1.03)",flush=True)
    with mlflow.start_run(run_name="optuna_best_xgb"):
        mlflow.log_params({**bp,"n":len(df),"n_trials":N_TRIALS})
        mlflow.log_metrics({"train_mae":mtr,"val_mae":mva,"test_mae":mte,"train_test_gap":gap})
    pd.DataFrame([{"trial":t.number,"val_mae":t.value,**t.params} for t in study.trials]).to_csv(f"{H}/viz_gxtb_20260625/optuna_xgb_trials.csv",index=False)
    import joblib; joblib.dump({"model":m,"scaler":sc,"features":F,"best_params":bp},f"{R}/pipeline/models/xgb_optuna_tuned_{pd.Timestamp.now():%Y%m%d}.joblib")
    print("DONE",flush=True)

if __name__=="__main__": main()
