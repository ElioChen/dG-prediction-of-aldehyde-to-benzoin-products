#!/usr/bin/env python
"""Unified FULL-DATASET benchmark of all Δ-correction models, logged to MLflow.
All runs on the SAME complete labels (dft_labels_all.parquet, ~219k) and the SAME 70/20/10
split (seed 42) -> finally apples-to-apples (earlier numbers were on partial labels / diff scripts).
Logs: baselines (raw g-xTB, GFN2) + Ridge/MLP/XGB/Ensemble × {34 prod, 56 +reactant, 72 +global},
per-scope MAE, + reference runs for Tier-1 and GNN dual_qm. Launch later: `mlflow ui --backend-store-uri <uri>`.
"""
import glob, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import mlflow
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
URI=f"sqlite:///{R}/mlflow_benchmark.db"   # MLflow 3.x deprecated the file store; use sqlite backend
mlflow.set_tracking_uri(URI); mlflow.set_experiment("gxtb_dft_full_benchmark_20260626")

PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new",
 "wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC",
 "dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int",
 "pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
 "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]
GKEYS=["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide",
       "has_P","has_B","has_S","has_Si","has_halogen"]; GLOB=[f"g_{k}" for k in GKEYS]
F34,F56,F72=PROD_QM, PROD_QM+ALDp, PROD_QM+ALDp+GLOB

def gfeats(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return {f"g_{k}":np.nan for k in GKEYS}
    s={a.GetSymbol() for a in m.GetAtoms()}
    v=[rdMolDescriptors.CalcTPSA(m),rdMolDescriptors.CalcNumHBD(m),rdMolDescriptors.CalcNumHBA(m),
       rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcFractionCSP3(m),
       rdMolDescriptors.CalcNumHeteroatoms(m),Descriptors.MolWt(m),rdMolDescriptors.CalcNumRings(m),
       rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcNumAliphaticRings(m),
       rdMolDescriptors.CalcNumAmideBonds(m),int('P'in s),int('B'in s),int('S'in s),int('Si'in s),
       int(bool(s&{'F','Cl','Br','I'}))]
    return {f"g_{k}":x for k,x in zip(GKEYS,v)}

def metrics(g,pred,y,cl):
    yh=g+pred; e=np.abs(yh-y)
    d={"test_mae":float(e.mean()),"test_rmse":float(np.sqrt(((yh-y)**2).mean())),
       "test_r2":float(1-((yh-y)**2).sum()/((y-y.mean())**2).sum())}
    for s in ["aromatic","aliphatic"]:
        mk=cl==s
        if mk.sum()>50: d[f"mae_{s}"]=float(e[mk].mean())
    return d

def main():
    dft=pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal","dG_xtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    # global (unique smiles)
    gc=f"{H}/viz_gxtb_20260625/global_descriptors_full.parquet"
    if Path(gc).exists(): g=pd.read_parquet(gc)
    else:
        u=df[["smiles"]].drop_duplicates(); g=pd.DataFrame([gfeats(s) for s in u.smiles]); g["smiles"]=u.smiles.values; g.to_parquet(gc,index=False)
    df=df.merge(g,on="smiles",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal"]+F72).reset_index(drop=True)
    df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    print("benchmark n",len(df),flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df)); tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    g0=df.dG_gxtb_kcal.values[te]; y=df.dG_orca_kcal.values[te]; cl=df.cls.values[te]; dtr=df.delta.values[tr]; dva=df.delta.values[va]

    def log_run(name,params,mets):
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params); mlflow.log_metrics(mets)
        print(f"  {name}: MAE={mets['test_mae']:.3f} R2={mets.get('test_r2',float('nan')):.3f}",flush=True)

    # baselines
    log_run("baseline_raw_gxtb",{"type":"baseline","n":len(df)},metrics(g0,np.zeros_like(g0),y,cl))
    log_run("baseline_raw_gfn2",{"type":"baseline","n":len(df)},
            {"test_mae":float(np.abs(df.dG_xtb_kcal.values[te]-y).mean())})

    def mk_models():
        return {"Ridge":lambda:Ridge(alpha=5.0),
                "MLP":lambda:MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10),
                "XGB":lambda:XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,n_jobs=24,early_stopping_rounds=60,eval_metric="mae")}
    for fname,F in [("f34_prod",F34),("f56_prod_react",F56),("f72_plus_global",F72)]:
        sc=StandardScaler().fit(df[F].values[tr]); Xtr,Xva,Xte=sc.transform(df[F].values[tr]),sc.transform(df[F].values[va]),sc.transform(df[F].values[te])
        preds={}
        for mn,mk in mk_models().items():
            m=mk()
            if mn=="XGB": m.fit(Xtr,dtr,eval_set=[(Xva,dva)],verbose=False)
            else: m.fit(Xtr,dtr)
            preds[mn]=m.predict(Xte)
            log_run(f"{mn}_{fname}",{"model":mn,"featset":fname,"nfeat":len(F),"n":len(df)},metrics(g0,preds[mn],y,cl))
        ens=0.5*preds["MLP"]+0.5*preds["XGB"]
        log_run(f"Ensemble_{fname}",{"model":"MLP+XGB_ens","featset":fname,"nfeat":len(F),"n":len(df)},metrics(g0,ens,y,cl))

    # reference runs (from prior GPU / different pipeline; tagged)
    for nm,mae,note in [("ref_GNN_dual_qm56",1.616,"GPU dual-encoder, n=202k"),
                        ("ref_GNN_gine_hybrid",2.13,"GPU graph+QM, partial n"),
                        ("ref_3D_DimeNet++",2.04,"60k subset, z+pos"),
                        ("ref_Tier1_pure_smiles",2.75,"Morgan+global, target dG direct")]:
        with mlflow.start_run(run_name=nm):
            mlflow.set_tag("source","reference_not_same_split"); mlflow.log_param("note",note); mlflow.log_metric("test_mae",mae)
    print(f"\nMLflow logged -> {URI}\n  view: mlflow ui --backend-store-uri {URI} --port 5000",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
