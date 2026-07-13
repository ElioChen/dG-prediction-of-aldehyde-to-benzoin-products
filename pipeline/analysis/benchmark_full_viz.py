#!/usr/bin/env python
"""Comprehensive benchmark with FULL metric logging (train/val/test MAE/RMSE/R²) + per-model
parity figures, all to MLflow (exp gxtb_dft_viz) + standalone PNGs. Annotates split sizes.
One standalone figure per model (no composites) + one comparison bar chart."""
import warnings, numpy as np, pandas as pd, mlflow
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore"); plt.rcParams.update({"figure.dpi":130,"savefig.bbox":"tight"})
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
FIG=Path(f"{H}/viz_gxtb_20260625/model_figures"); FIG.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{R}/mlflow_benchmark.db"); mlflow.set_experiment("gxtb_dft_viz_fullmetrics")
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole","mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; GLOB=[f"g_{k}" for k in ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide","has_P","has_B","has_S","has_Si","has_halogen"]]
F72=PROD_QM+ALDp+GLOB

def mm(g,pred,y):
    yh=g+pred; e=yh-y
    return dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),r2=float(1-(e**2).sum()/((y-y.mean())**2).sum()))

def main():
    dft=pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    g=pd.read_parquet(f"{H}/viz_gxtb_20260625/global_descriptors_full.parquet")
    df=p.merge(dft,on="id");df["ald_id"]=df.donor_id.astype("Int64");df=df.merge(a,on="ald_id",how="left").merge(g,on="smiles",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+F72);df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True);df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    i=np.random.default_rng(42).permutation(len(df));ntr,nva=int(.7*len(df)),int(.9*len(df));tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    print(f"split train {len(tr)} val {len(va)} test {len(te)}",flush=True)
    sc=StandardScaler().fit(df[F72].values[tr]); Xtr,Xva,Xte=sc.transform(df[F72].values[tr]),sc.transform(df[F72].values[va]),sc.transform(df[F72].values[te])
    G={s:df.dG_gxtb_kcal.values[ix] for s,ix in [("tr",tr),("va",va),("te",te)]}; Y={s:df.dG_orca_kcal.values[ix] for s,ix in [("tr",tr),("va",va),("te",te)]}
    dtr,dva=df.delta.values[tr],df.delta.values[va]
    summary=[]
    def run(name,fit_pred):
        ptr,pva,pte=fit_pred()
        mtr,mva,mte=mm(G["tr"],ptr,Y["tr"]),mm(G["va"],pva,Y["va"]),mm(G["te"],pte,Y["te"])
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model":name,"nfeat":len(F72),"n_train":len(tr),"n_val":len(va),"n_test":len(te)})
            for sp,m in [("train",mtr),("val",mva),("test",mte)]:
                mlflow.log_metrics({f"{sp}_mae":m["mae"],f"{sp}_rmse":m["rmse"],f"{sp}_r2":m["r2"]})
        # parity figure (test)
        yh=G["te"]+pte; fig,ax=plt.subplots(figsize=(5,5)); ax.hexbin(Y["te"],yh,gridsize=60,bins="log",cmap="viridis",mincnt=1,extent=(-25,30,-25,30))
        ax.plot([-25,30],[-25,30],"w--",lw=1);ax.set_xlim(-25,30);ax.set_ylim(-25,30)
        ax.set_xlabel("DFT ΔG (kcal/mol)");ax.set_ylabel(f"{name} pred");ax.set_title(f"{name}  test MAE={mte['mae']:.3f} R²={mte['r2']:.3f}\ntrain {mtr['mae']:.2f} | val {mva['mae']:.2f} | test {mte['mae']:.2f}  (n={len(tr)}/{len(va)}/{len(te)})")
        fig.savefig(FIG/f"parity_{name}.png");plt.close(fig)
        summary.append({"model":name,"train_mae":mtr["mae"],"val_mae":mva["mae"],"test_mae":mte["mae"],"test_r2":mte["r2"]})
        print(f"  {name}: train {mtr['mae']:.3f} val {mva['mae']:.3f} test {mte['mae']:.3f}",flush=True)
    def lin(): m=Ridge(1.0).fit(Xtr,dtr); return m.predict(Xtr),m.predict(Xva),m.predict(Xte)
    run("Ridge72",lin)
    for hl in [(256,128),(512,256,128)]:
        def f(hl=hl): m=MLPRegressor(hidden_layer_sizes=hl,alpha=1e-4,max_iter=250,early_stopping=True,n_iter_no_change=12).fit(Xtr,dtr); return m.predict(Xtr),m.predict(Xva),m.predict(Xte)
        run(f"MLP_{'-'.join(map(str,hl))}",f)
    xgbs={}
    for d,ne in [(8,1500),(10,2000)]:
        def f(d=d,ne=ne):
            m=XGBRegressor(n_estimators=ne,max_depth=d,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,n_jobs=24,early_stopping_rounds=60,eval_metric="mae").fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
            xgbs[d]=m; return m.predict(Xtr),m.predict(Xva),m.predict(Xte)
        run(f"XGB_d{d}",f)
    mlp=MLPRegressor(hidden_layer_sizes=(512,256,128),alpha=1e-4,max_iter=250,early_stopping=True,n_iter_no_change=12).fit(Xtr,dtr)
    def ens():
        pr=lambda S:(mlp.predict(S)+xgbs[8].predict(S)+xgbs[10].predict(S))/3
        return pr(Xtr),pr(Xva),pr(Xte)
    run("Ensemble_MLP+XGB8+XGB10",ens)
    s=pd.DataFrame(summary).sort_values("test_mae"); s.to_csv(FIG/"summary_train_val_test.csv",index=False)
    # comparison bar (test) with train overlay
    fig,ax=plt.subplots(figsize=(8,4.5)); s2=s.iloc[::-1]
    ax.barh(s2.model,s2.test_mae,color="#3690c0",label="test"); ax.barh(s2.model,s2.train_mae,height=0.4,color="#f16913",label="train")
    ax.set_xlabel("MAE (kcal/mol)");ax.set_title("All models: train vs test MAE (full 219k, 72 feat)");ax.legend()
    fig.savefig(FIG/"all_models_train_vs_test.png");plt.close(fig)
    print(f"\nfigures -> {FIG}",flush=True);print("DONE",flush=True)

if __name__=="__main__": main()
