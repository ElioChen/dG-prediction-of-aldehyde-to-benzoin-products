#!/usr/bin/env python
"""[2] Scaffold-split generalization: train 72-feat ensemble with Murcko-scaffold-disjoint
train/test (honest extrapolation to novel cores) vs random split. Logs to MLflow."""
import glob, warnings, numpy as np, pandas as pd, mlflow
from collections import defaultdict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
mlflow.set_tracking_uri(f"sqlite:///{R}/mlflow_benchmark.db"); mlflow.set_experiment("exp2_scaffold_split")
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
    print("scaffolding",len(df),flush=True)
    def scaf(s):
        try: return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        except Exception: return s
    df["scaf"]=[scaf(s) for s in df.smiles]
    def fit_eval(tr,te,tag):
        sc=StandardScaler().fit(df[FEATS].values[tr]); Xtr,Xte=sc.transform(df[FEATS].values[tr]),sc.transform(df[FEATS].values[te])
        ml=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10).fit(Xtr,df.delta.values[tr])
        gb=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,n_jobs=24,eval_metric="mae").fit(Xtr,df.delta.values[tr])
        pred=0.5*ml.predict(Xte)+0.5*gb.predict(Xte); yh=df.dG_gxtb_kcal.values[te]+pred; y=df.dG_orca_kcal.values[te]
        mae=float(np.abs(yh-y).mean()); r2=float(1-((yh-y)**2).sum()/((y-y.mean())**2).sum())
        with mlflow.start_run(run_name=tag): mlflow.log_params({"split":tag,"n":len(df),"nfeat":len(FEATS)}); mlflow.log_metrics({"test_mae":mae,"test_r2":r2})
        print(f"  {tag}: MAE={mae:.3f} R2={r2:.3f}",flush=True); return mae
    rng=np.random.default_rng(42); i=rng.permutation(len(df)); nt=int(.9*len(df))
    rnd=fit_eval(i[:nt],i[nt:],"random_split")
    # scaffold split: whole scaffolds to test (~10%)
    groups=defaultdict(list); [groups[s].append(k) for k,s in enumerate(df.scaf)]
    order=list(groups); rng.shuffle(order); te=[];
    for s in order:
        if len(te)<0.1*len(df): te+=groups[s]
        else: break
    teset=set(te); tr=[k for k in range(len(df)) if k not in teset]
    scf=fit_eval(np.array(tr),np.array(te),"scaffold_split")
    print(f"\nrandom {rnd:.3f} -> scaffold {scf:.3f}  (Δ {scf-rnd:+.3f}, generalization gap)",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
