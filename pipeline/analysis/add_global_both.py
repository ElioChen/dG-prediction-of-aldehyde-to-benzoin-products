#!/usr/bin/env python
"""Global (RDKit 2D) descriptors on BOTH product and reactant(aldehyde). Compare
56 / 56+prod-global / 56+prod+react-global. Reactant global computed on unique aldehydes.
"""
import glob, warnings, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings("ignore")
R="/scratch-shared/schen3/benzoin-dg"; H=f"{R}/data/cross_benzoin/homo_v6"; OUT=f"{H}/viz_gxtb_20260625"
PROD_QM=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH","wbo_CO_ket","wbo_CC_new",
 "wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC","fukui_plus_carbC","fukui_minus_carbC",
 "dual_carbC","vbur_ketC","vbur_carbC","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int",
 "pa_ketO","hb_dist","hb_angle","dih_core"]
ALD=["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
 "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C","wbo_CO",
 "pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp=[f"ald_{c}" for c in ALD]; FEATS=PROD_QM+ALDp
GKEYS=["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing","nAmide",
       "has_P","has_B","has_S","has_Si","has_halogen"]
def gfeats(smi,pre):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return {f"{pre}_{k}":np.nan for k in GKEYS}
    s={a.GetSymbol() for a in m.GetAtoms()}
    v=[rdMolDescriptors.CalcTPSA(m),rdMolDescriptors.CalcNumHBD(m),rdMolDescriptors.CalcNumHBA(m),
       rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcFractionCSP3(m),
       rdMolDescriptors.CalcNumHeteroatoms(m),Descriptors.MolWt(m),rdMolDescriptors.CalcNumRings(m),
       rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcNumAliphaticRings(m),
       rdMolDescriptors.CalcNumAmideBonds(m),int('P' in s),int('B' in s),int('S' in s),int('Si' in s),
       int(bool(s&{'F','Cl','Br','I'}))]
    return {f"{pre}_{k}":x for k,x in zip(GKEYS,v)}
PGLOB=[f"pg_{k}" for k in GKEYS]; AGLOB=[f"ag_{k}" for k in GKEYS]

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","donor_smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal","smiles","donor_smiles"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    # product global — compute fresh on UNIQUE product smiles, then map (avoids cache/column clashes)
    t0=time.time(); up=df[["smiles"]].drop_duplicates()
    pg=pd.DataFrame([gfeats(s,"pg") for s in up.smiles]); pg["smiles"]=up.smiles.values
    df=df.merge(pg,on="smiles",how="left")
    print(f"product global on {len(up)} unique products ({time.time()-t0:.0f}s)",flush=True)
    # reactant global on UNIQUE aldehydes (fast), then map
    t0=time.time(); uniq=df[["ald_id","donor_smiles"]].drop_duplicates("ald_id")
    ag=pd.DataFrame([gfeats(s,"ag") for s in uniq.donor_smiles]); ag["ald_id"]=uniq.ald_id.values
    df=df.merge(ag,on="ald_id",how="left")
    print(f"reactant global on {len(uniq)} unique aldehydes ({time.time()-t0:.0f}s)",flush=True)
    df=df.dropna(subset=PGLOB+AGLOB).reset_index(drop=True)
    print("n",len(df),flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df)); tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    def fit(cols):
        sc=StandardScaler().fit(df[cols].values[tr])
        m=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,
            min_child_weight=5,n_jobs=16,early_stopping_rounds=60,eval_metric="mae").fit(
            sc.transform(df[cols].values[tr]),df.delta.values[tr],eval_set=[(sc.transform(df[cols].values[va]),df.delta.values[va])],verbose=False)
        g0=df.dG_gxtb_kcal.values[te]; y=df.dG_orca_kcal.values[te]; err=np.abs(g0+m.predict(sc.transform(df[cols].values[te]))-y); cl=df.cls.values[te]
        r={"all":err.mean(),"aromatic":err[cl=='aromatic'].mean(),"aliphatic":err[cl=='aliphatic'].mean()}
        e=err[cl=='aliphatic']; r["ali_tail15"]=e[e>=np.quantile(e,.85)].mean(); return r
    out={}
    for nm,cols in [("56",FEATS),("56+prodG",FEATS+PGLOB),("56+prod+reactG",FEATS+PGLOB+AGLOB)]:
        out[nm]=fit(cols); print(nm,{k:round(v,3) for k,v in out[nm].items()},flush=True)
    pd.DataFrame(out).T.to_csv(f"{OUT}/global_both_ablation.csv")
    print("DONE",flush=True)

if __name__=="__main__": main()
