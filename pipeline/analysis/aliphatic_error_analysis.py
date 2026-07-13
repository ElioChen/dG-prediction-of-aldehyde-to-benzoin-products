#!/usr/bin/env python
"""Where does the model fail on ALIPHATIC aldehydes, and what descriptor is missing?
Train champion XGB (56 feat), take aliphatic test residuals, compare HIGH-error vs LOW-error
subsets by (a) functional-group SMARTS enrichment on the aldehyde, (b) which of the 56
descriptors differ most. Enriched motifs absent from the descriptor set => candidate new features.
"""
import glob, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
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

# functional-group SMARTS evaluated on the aldehyde (donor_smiles)
SMARTS={
 "alpha_branched(3deg)":"[CX4H1,CX4H0]([#6])([#6])[CX3H1]=O",
 "alpha_quaternary":"[CX4H0]([#6])([#6])([#6])[CX3H1]=O",
 "alpha_O":"[#8][CX4][CX3H1]=O",
 "alpha_N":"[#7][CX4][CX3H1]=O",
 "alpha_halogen":"[F,Cl,Br,I][CX4][CX3H1]=O",
 "alpha_S":"[#16][CX4][CX3H1]=O",
 "beta_branch":"[CX4]([#6])([#6])[CX4][CX3H1]=O",
 "CC_double_bond":"[CX3]=[CX3]",
 "CC_triple":"C#C",
 "nitrile":"C#N",
 "ester":"[CX3](=O)[OX2H0]",
 "amide":"[CX3](=O)[NX3]",
 "ketone_extra":"[#6][CX3](=O)[#6]",
 "ether":"[OD2]([#6])[#6]",
 "hydroxyl":"[OX2H]",
 "sulfur_any":"[#16]",
 "fluorine_any":"[F]",
 "Si":"[Si]","B":"[B]","P":"[P]",
 "ring_aliphatic":"[R;CX4]",
 "cyclopropyl":"[CX4;r3]",
 "tertiary_amine":"[NX3]([#6])([#6])[#6]",
}
PAT={k:Chem.MolFromSmarts(v) for k,v in SMARTS.items()}

def feats_of(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    d={k:int(m.HasSubstructMatch(p)) for k,p in PAT.items() if p is not None}
    d["heavy_atoms"]=m.GetNumHeavyAtoms()
    d["n_rings"]=rdMolDescriptors.CalcNumRings(m)
    d["n_rotbonds"]=rdMolDescriptors.CalcNumRotatableBonds(m)
    return d

def main():
    fs=sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv"))
    dft=pd.concat([pd.read_csv(f,usecols=["id","dG_orca_kcal"]) for f in fs],ignore_index=True).dropna().drop_duplicates("id")
    p=pd.read_csv(f"{H}/products_all.csv",usecols=["id","donor_id","donor_smiles","dG_gxtb_kcal"]+PROD_QM,low_memory=False)
    a=pd.read_csv(f"{H}/aldehydes_all.csv",usecols=["id"]+ALD,low_memory=False).drop_duplicates("id").rename(columns={"id":"ald_id",**{c:f"ald_{c}" for c in ALD}})
    cls=pd.read_parquet(f"{H}/aldehyde_class.parquet")
    df=p.merge(dft,on="id"); df["ald_id"]=df.donor_id.astype("Int64"); df=df.merge(a,on="ald_id",how="left").merge(cls,on="id",how="left")
    df=df.dropna(subset=["dG_gxtb_kcal","dG_orca_kcal","donor_smiles"]+FEATS); df=df[df.dG_orca_kcal.abs()<60].reset_index(drop=True)
    df["delta"]=df.dG_orca_kcal-df.dG_gxtb_kcal
    i=np.random.default_rng(42).permutation(len(df)); ntr,nva=int(.7*len(df)),int(.9*len(df))
    tr,va,te=i[:ntr],i[ntr:nva],i[nva:]
    sc=StandardScaler().fit(df[FEATS].values[tr])
    m=XGBRegressor(n_estimators=800,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,
        n_jobs=16,early_stopping_rounds=50,eval_metric="mae").fit(sc.transform(df[FEATS].values[tr]),df.delta.values[tr],
        eval_set=[(sc.transform(df[FEATS].values[va]),df.delta.values[va])],verbose=False)
    test=df.iloc[te].copy()
    test["err"]=np.abs(test.dG_gxtb_kcal.values+m.predict(sc.transform(test[FEATS].values))-test.dG_orca_kcal.values)
    ali=test[test.cls=="aliphatic"].copy()
    print(f"aliphatic test n={len(ali)}  mean|err|={ali.err.mean():.3f}",flush=True)
    hi=ali[ali.err>=ali.err.quantile(.85)]; lo=ali[ali.err<=ali.err.quantile(.50)]
    print(f"HIGH-error (top15%) n={len(hi)} mean={hi.err.mean():.2f} | LOW (bottom50%) n={len(lo)} mean={lo.err.mean():.2f}\n",flush=True)
    # functional-group enrichment
    fh=pd.DataFrame([feats_of(s) for s in hi.donor_smiles]).mean()
    fl=pd.DataFrame([feats_of(s) for s in lo.donor_smiles]).mean()
    enr=pd.DataFrame({"high":fh,"low":fl}); enr["ratio"]=(enr.high+1e-3)/(enr.low+1e-3)
    print("=== functional-group / structural enrichment (high vs low error) ===",flush=True)
    print(enr.sort_values("ratio",ascending=False).head(14).round(3).to_string(),flush=True)
    # descriptor gaps (which of 56 differ most, standardized)
    z=(df[FEATS]-df[FEATS].mean())/df[FEATS].std()
    zhi=z.loc[hi.index].mean(); zlo=z.loc[lo.index].mean(); gap=(zhi-zlo).abs().sort_values(ascending=False)
    print("\n=== descriptors most different (high vs low, |Δ std|) ===",flush=True)
    print(gap.head(12).round(3).to_string(),flush=True)
    # worst examples
    worst=ali.sort_values("err",ascending=False).head(15)[["donor_smiles","dG_gxtb_kcal","dG_orca_kcal","err"]]
    worst.to_csv(f"{OUT}/aliphatic_worst_examples.csv",index=False)
    enr.sort_values("ratio",ascending=False).to_csv(f"{OUT}/aliphatic_error_enrichment.csv")
    print("\n=== 15 worst aliphatic ===\n"+worst.round(2).to_string(index=False),flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
