"""[4] Tier-1 distillation: train pure-SMILES student (Morgan2048+16 global -> ΔG) on
(a) hard DFT labels vs (b) Tier-2 ensemble's predicted ΔG (soft teacher). Eval both on a
held-out test set against true DFT. Tests whether teacher-denoised targets give a better Tier-1.
Logs to MLflow."""
import warnings, time, numpy as np, pandas as pd, mlflow
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"
mlflow.set_tracking_uri(f"sqlite:///{R}/mlflow_benchmark.db"); mlflow.set_experiment("exp4_tier1_distill")
GK=["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide","has_P","has_B","has_S","has_Si","has_halogen"]; NB=2048

def feat(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    arr=np.zeros(NB,dtype=np.int8); ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=NB),arr)
    s={a.GetSymbol() for a in m.GetAtoms()}
    g=[rdMolDescriptors.CalcTPSA(m),rdMolDescriptors.CalcNumHBD(m),rdMolDescriptors.CalcNumHBA(m),rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcFractionCSP3(m),rdMolDescriptors.CalcNumHeteroatoms(m),Descriptors.MolWt(m),rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcNumAliphaticRings(m),rdMolDescriptors.CalcNumAmideBonds(m),int('P'in s),int('B'in s),int('S'in s),int('Si'in s),int(bool(s&{'F','Cl','Br','I'}))]
    return np.concatenate([arr.astype(np.float32),np.array(g,dtype=np.float32)])

def main():
    dft=pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","smiles"],low_memory=False)
    teach=pd.read_csv(f"{H}/viz_gxtb_20260625/products_dG_corrected_FINAL_20260626.csv",usecols=["id","dG_gxtb_corrected_final"])
    df=p.merge(dft,on="id").merge(teach,on="id",how="left").dropna(subset=["smiles","dG_orca_kcal","dG_gxtb_corrected_final"])
    df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    print("n",len(df),flush=True); t0=time.time(); X=[]; keep=[]
    for k,s in enumerate(df.smiles):
        f=feat(s)
        if f is not None: X.append(f); keep.append(k)
        if k%40000==0: print(f"  feat {k} ({time.time()-t0:.0f}s)",flush=True)
    X=np.vstack(X); df=df.iloc[keep].reset_index(drop=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df)); tr,te=i[:ntr],i[ntr:]
    yor=df.dG_orca_kcal.values; ytea=df.dG_gxtb_corrected_final.values
    def fit(target,tag):
        m=XGBRegressor(n_estimators=1200,max_depth=8,learning_rate=0.03,subsample=0.8,colsample_bytree=0.6,n_jobs=24,eval_metric="mae").fit(X[tr],target[tr])
        yp=m.predict(X[te]); mae=float(np.abs(yp-yor[te]).mean()); r2=float(1-((yp-yor[te])**2).sum()/((yor[te]-yor[te].mean())**2).sum())
        with mlflow.start_run(run_name=tag): mlflow.log_params({"target":tag,"n":len(df)}); mlflow.log_metrics({"test_mae_vs_dft":mae,"test_r2":r2})
        print(f"  {tag}: test MAE(vs DFT)={mae:.3f} R2={r2:.3f}",flush=True); return mae
    h=fit(yor,"tier1_hard_DFT"); s=fit(ytea,"tier1_soft_teacher")
    print(f"\nhard {h:.3f} vs soft-teacher {s:.3f} (Δ {s-h:+.3f})",flush=True); print("DONE",flush=True)

if __name__=="__main__": main()
