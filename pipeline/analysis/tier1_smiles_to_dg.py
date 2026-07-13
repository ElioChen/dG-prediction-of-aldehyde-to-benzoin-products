#!/usr/bin/env python
"""TIER-1 triage model: pure SMILES -> ΔG (DFT r2SCAN-3c), NO xTB/QM front-end.
Features = Morgan fingerprint (r2, 2048b) + 16 RDKit 2D global descriptors -- all instant.
Target = dG_orca directly (gold label). For ms/molecule screening of millions.
Compares to the Tier-2 accurate model (1.61, needs xTB+QM). CPU (gnn env: rdkit+xgboost).
"""
import glob, warnings, time, numpy as np, pandas as pd, joblib, json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"; OUT=Path(f"{H}/viz_gxtb_20260625")
MODELDIR=Path(f"{R}/pipeline/models"); TAG=time.strftime("%Y%m%d")
GKEYS=["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide",
       "has_P","has_B","has_S","has_Si","has_halogen"]
NB=2048

def featurize(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    fp=np.zeros(NB,dtype=np.float32)
    AllChem.GetHashedMorganFingerprint  # ensure import
    bv=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=NB)
    from rdkit.DataStructs import ConvertToNumpyArray; arr=np.zeros(NB,dtype=np.int8); ConvertToNumpyArray(bv,arr)
    s={a.GetSymbol() for a in m.GetAtoms()}
    g=[rdMolDescriptors.CalcTPSA(m),rdMolDescriptors.CalcNumHBD(m),rdMolDescriptors.CalcNumHBA(m),
       rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcFractionCSP3(m),
       rdMolDescriptors.CalcNumHeteroatoms(m),Descriptors.MolWt(m),rdMolDescriptors.CalcNumRings(m),
       rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcNumAliphaticRings(m),
       rdMolDescriptors.CalcNumAmideBonds(m),int('P'in s),int('B'in s),int('S'in s),int('Si'in s),
       int(bool(s&{'F','Cl','Br','I'}))]
    return np.concatenate([arr.astype(np.float32),np.array(g,dtype=np.float32)])

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))+sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/retry7200/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id",keep="last")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","smiles"],low_memory=False)
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id").merge(cls,on="id",how="left")
    df=df[df.dG_orca_kcal.abs()<60].dropna(subset=["smiles"]).reset_index(drop=True)
    print("n labeled",len(df),flush=True)
    t0=time.time(); X=[]; keep=[]
    for k,s in enumerate(df.smiles):
        f=featurize(s)
        if f is not None: X.append(f); keep.append(k)
        if k%40000==0: print(f"  feat {k} ({time.time()-t0:.0f}s)",flush=True)
    X=np.vstack(X); df=df.iloc[keep].reset_index(drop=True); y=df.dG_orca_kcal.values
    print("X",X.shape,flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df)); tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    m=XGBRegressor(n_estimators=1200,max_depth=8,learning_rate=0.03,subsample=0.8,colsample_bytree=0.6,
        n_jobs=24,early_stopping_rounds=50,eval_metric="mae").fit(X[tr],y[tr],eval_set=[(X[va],y[va])],verbose=False)
    yp=m.predict(X[te]); err=np.abs(yp-y[te]); cl=df.cls.values[te]
    mae=float(err.mean()); r2=float(1-((yp-y[te])**2).sum()/((y[te]-y[te].mean())**2).sum())
    print(f"\nTIER-1 (pure SMILES, Morgan+global -> DFT ΔG) TEST MAE={mae:.3f} R2={r2:.3f}",flush=True)
    for s in ["aromatic","aliphatic"]:
        mk=cl==s; print(f"  {s}: MAE={err[mk].mean():.3f}",flush=True)
    print(f"ref Tier-2 (xTB+QM ensemble): MAE 1.61",flush=True)
    joblib.dump({"model":m,"nbits":NB,"gkeys":GKEYS,"target":"DFT_r2scan3c ΔG (kcal/mol)","test_mae":mae},
                MODELDIR/f"tier1_smiles_to_dG_{TAG}.joblib")
    json.dump({"model":"XGB Morgan2048+16global","target":"dG_orca direct","test_mae":mae,"test_r2":r2,"n":len(df)},
              open(OUT/f"tier1_results_{TAG}.json","w"),indent=2)
    print("DONE",flush=True)

if __name__=="__main__": main()
