#!/usr/bin/env python
"""Add whole-molecule (global) descriptors targeting the aliphatic high-error tail
(large/flexible/polar/P-B-S molecules). Compare 56 vs 56+global on all/aromatic/aliphatic
and on the aliphatic high-error 15% tail.  Global feats are instant RDKit on product SMILES.
"""
import glob, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
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
GLOBAL=["g_TPSA","g_HBD","g_HBA","g_RotB","g_FracCsp3","g_nHetero","g_MolWt","g_nRing",
        "g_nAromRing","g_nAliphRing","g_nAmide","g_has_P","g_has_B","g_has_S","g_has_Si","g_has_halogen"]
GCACHE=f"{OUT}/global_descriptors.parquet"

def gfeats(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    syms={a.GetSymbol() for a in m.GetAtoms()}
    return dict(g_TPSA=rdMolDescriptors.CalcTPSA(m),g_HBD=rdMolDescriptors.CalcNumHBD(m),
        g_HBA=rdMolDescriptors.CalcNumHBA(m),g_RotB=rdMolDescriptors.CalcNumRotatableBonds(m),
        g_FracCsp3=rdMolDescriptors.CalcFractionCSP3(m),g_nHetero=rdMolDescriptors.CalcNumHeteroatoms(m),
        g_MolWt=Descriptors.MolWt(m),g_nRing=rdMolDescriptors.CalcNumRings(m),
        g_nAromRing=rdMolDescriptors.CalcNumAromaticRings(m),g_nAliphRing=rdMolDescriptors.CalcNumAliphaticRings(m),
        g_nAmide=rdMolDescriptors.CalcNumAmideBonds(m),
        g_has_P=int('P' in syms),g_has_B=int('B' in syms),g_has_S=int('S' in syms),
        g_has_Si=int('Si' in syms),g_has_halogen=int(bool(syms&{'F','Cl','Br','I'})))

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal","smiles"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    # global descriptors (cache)
    if Path(GCACHE).exists():
        g=pd.read_parquet(GCACHE)
    else:
        import time; t0=time.time(); rows=[]
        for k,(i,s) in enumerate(zip(df.id,df.smiles)):
            d=gfeats(s); d=d or {c:np.nan for c in GLOBAL}; d["id"]=i; rows.append(d)
            if k%40000==0: print(f"  global {k} ({time.time()-t0:.0f}s)",flush=True)
        g=pd.DataFrame(rows); g.to_parquet(GCACHE,index=False)
    df=df.merge(g,on="id",how="left").dropna(subset=GLOBAL)
    print("n",len(df),flush=True)
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df))
    tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    def fit(cols):
        sc=StandardScaler().fit(df[cols].values[tr])
        m=XGBRegressor(n_estimators=1500,max_depth=8,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,
            min_child_weight=5,n_jobs=16,early_stopping_rounds=60,eval_metric="mae").fit(
            sc.transform(df[cols].values[tr]),df.delta.values[tr],
            eval_set=[(sc.transform(df[cols].values[va]),df.delta.values[va])],verbose=False)
        g0=df.dG_gxtb_kcal.values[te]; y=df.dG_orca_kcal.values[te]; err=np.abs(g0+m.predict(sc.transform(df[cols].values[te]))-y)
        cl=df.cls.values[te]
        res={"all":err.mean()}
        for s in ["aromatic","aliphatic"]:
            mk=cl==s; res[s]=err[mk].mean()
            if s=="aliphatic":
                e=err[mk]; res["aliphatic_tail15%"]=e[e>=np.quantile(e,.85)].mean()
        return res,m,sc,cols
    print("=== 56 features ===",flush=True); r56,_,_,_=fit(FEATS); print(r56,flush=True)
    print("=== 56 + global ===",flush=True); rg,m,sc,cols=fit(FEATS+GLOBAL); print(rg,flush=True)
    # importance of the new global feats
    imp=pd.Series(m.feature_importances_,index=cols).loc[GLOBAL].sort_values(ascending=False)
    print("\n=== global-feature importances ===\n"+imp.round(4).to_string(),flush=True)
    print("\n=== Δ (56 -> 56+global) ===",flush=True)
    for k in ["all","aromatic","aliphatic","aliphatic_tail15%"]:
        print(f"  {k:18s} {r56.get(k,float('nan')):.3f} -> {rg.get(k,float('nan')):.3f}  ({rg.get(k,0)-r56.get(k,0):+.3f})",flush=True)
    pd.DataFrame([dict(featset="56",**r56),dict(featset="56+global",**rg)]).to_csv(f"{OUT}/global_descriptor_ablation.csv",index=False)
    print("DONE",flush=True)

if __name__=="__main__": main()
