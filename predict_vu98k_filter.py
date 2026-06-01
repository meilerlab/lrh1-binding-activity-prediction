#!/usr/bin/env python3

'''
predict_vu98k_filter.py: uses BLiP-L and ALiP-L to predict binder, activity for VU98k compounds. 
Filters list to predicted binders and activators. Outputs compounds and metrics to terminal or CSV file. 

To execute: python predict_vu98k_filter.py

User needs to set the variable location_of_all_docks
'''

import sys
import os
import torch
import pandas as pd 
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator, Draw

# ANN architecture and data format 
import utils_drugfilters as drugfilters
from utils_define_data_model import BioData,Network,NetworkActivity
import predict

##############################
# User-specified location of all docked ligands from screen 
location_of_all_docks='~/ulls/vulibrary/' # location of all docked poses 

##############################
# comparison compounds
rjw100='CCCCCCC1=C([C@@]2(CC[C@H]([C@@H]2C1)O)C(=C)c3ccccc3)c4ccccc4'
iuw = 'C=C(C1=CC=CC=C1)[C@@]23CC[C@@H]([C@@H]2CC(=C3C4=CC=CC=C4)CCCCCCCCCC(=O)O)NS(=O)(=O)N' # 6n10ca
ifname_4504='~/spectrum2k/rldock/s2k_33/s2k_33_0025/s2k_33_0025.sdf' # 4504 from spectrum2k
ifname_9647='~/spectrum2k/rldock/s2k_34/s2k_34_0011/s2k_34_0011.sdf' # 9647 from spectrum2k
smi_4504=Chem.MolToSmiles(Chem.SDMolSupplier(ifname_4504)[0])        
smi_9647=Chem.MolToSmiles(Chem.SDMolSupplier(ifname_9647)[0])
##############################

def main():

    eterms_list=['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol','if_X_hbond_bb_sc','if_X_hbond_sc']

    # load saved models from train_crossval_save.py 
    model_blipl = 'blip-l.pth'   
    model_alipl = 'alip-l.pth'
  
    # Load energy terms for 98k compounds 
    rlfullDF=pd.read_pickle('../data/eterms_vu98k_iuw_full.pkl')        
    rl4pldDF=pd.read_pickle('../data/eterms_vu98k_iuw_4pld.pkl')
    rl7tt8DF=pd.read_pickle('../data/eterms_vu98k_iuw_7tt8.pkl')

    # Get binder and activity predictions 
    rlfullDF = getPreds(eterms_list.copy(),model_blipl,model_alipl,rlfullDF)
    rl4pldDF = getPreds(eterms_list.copy(),model_blipl,model_alipl,rl4pldDF)
    rl7tt8DF = getPreds(eterms_list.copy(),model_blipl,model_alipl,rl7tt8DF)

    # Combine dataframes, save one entry per compound with lowest IDX to bestIDXDF    
    combinedDF = pd.concat([rlfullDF, rl4pldDF, rl7tt8DF])
    #dumdf = combinedDF.copy()
    #dumdf['sigmoid_binder_pred'] = 1 / (1 + np.exp(-dumdf['binder_pred']))
    #dumdf['sigmoid_activity_pred'] = 1 / (1 + np.exp(-dumdf['activity_pred']))
    #dumdf.to_csv(f'../scripts/temp_out_predict_vu98k_all_preds.csv',index=False,float_format="%.2f")
    #quit()
    
    sortDF = combinedDF.sort_values(by=['ID','interface_delta_X'])
    bestIDXDF0 = sortDF.groupby('ID').first().reset_index()

    # Logits to sigmoid to report probs 
    bestIDXDF = bestIDXDF0.copy()
    bestIDXDF['sigmoid_binder_pred'] = 1 / (1 + np.exp(-bestIDXDF['binder_pred']))
    bestIDXDF['sigmoid_activity_pred'] = 1 / (1 + np.exp(-bestIDXDF['activity_pred']))
    bestIDXDF.to_csv(f'./out/list_all_vu98k.csv',index=False,float_format="%.2f")        
    
    ##########
    # Filter compounds     
    s1DF, s2DF = runFilters(bestIDXDF) 
    filterDFs = [s1DF, s2DF]

    # Filter compounds, just 7tt8 set 
    s1DF, s2DF = runFilters(pd.concat([rl7tt8DF]))
    filterDFs_7tt8 = [s1DF, s2DF]

    combinedFilterDF = pd.concat(filterDFs + filterDFs_7tt8)
    combinedFilterDF = combinedFilterDF.drop_duplicates(subset=['ID', 'posepath'])

    combinedFilterDF_excl_nondruglike = getProperties(combinedFilterDF,exclude_nondruglike=True,nFilters=2)
    combinedFilterDF_excl_nondruglike.to_csv(f'./out/list_filter.csv',index=False,float_format="%.2f")

    return 

##############################
def getPreds(eterms,modelBinder,modelActivity,df):
    ''' 
    Get binding and activity prediction. 
    '''

    # program exits if all eterms are not present in dataframe 
    all_eterms_present = set(eterms).issubset(df.columns)
    if not all_eterms_present:
        sys.exit("Error: Required energy term columns are not all listed in inputCSV. See eterms_list for required eterms.")
    
    features_array = df[eterms].to_numpy()
    with torch.inference_mode():
        binder_pred , _ = predict.runTestSet(Network,modelBinder, features_array)

    df['binder_pred'] = binder_pred
    
    eterms.append('binder_pred')
    features_array = df[eterms].to_numpy()
    with torch.inference_mode():
        activity_pred , _ = predict.runTestSet(NetworkActivity,modelActivity, features_array)

    df['activity_pred'] = activity_pred 

    return df 

##############################
def runFilters(df0):
    '''
    Executes cutoff and count filter to narrow list of compounds. 
    '''

    # recall preds are in logits, so range is (-inf,inf) and predicted hits should have pred value > 0

    if ('sigmoid_binder_pred' or 'sigmoid_activity_pred') not in df0.columns:
        df = df0.copy()
        df['sigmoid_binder_pred'] = 1 / (1 + np.exp(-df['binder_pred']))
        df['sigmoid_activity_pred'] = 1 / (1 + np.exp(-df['activity_pred']))
    else:
        df = df0.copy()    
    
    predHitDF = df[(df['sigmoid_binder_pred'] > 0.6) ] 
    filter1DF = predHitDF.nlargest(8000,'activity_pred')
    filter2DF = predHitDF.nsmallest(4000,'interface_delta_X')

    return filter1DF, filter2DF

##############################
def addSMILES(dir_all_ligands,dir_this_lig): 
    # get smiles string of ligand belonging to dockdir

    try:
        ligdir = os.path.dirname(dir_this_lig).split(os.sep)
        sdf = os.path.join(dir_all_ligands,ligdir[0],ligdir[1],f"{ligdir[1]}.sdf")
        if not os.path.exists(sdf):
            return None
        mol = Chem.SDMolSupplier(sdf)[0]
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None 

##############################
def compute_properties(smiles):
    # Compute molecular properties from smiles
    
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        logp = Crippen.MolLogP(mol)
        return mw, heavy_atoms, logp
    return None, None  # Handle invalid SMILES

##############################
def excludeNonDrugLike(df0,nFilters):
    # Excludes non drug-like compounds. User can set the number of filters they want to consider

    mask = []
    
    for _, row in df0.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        matchList = drugfilters.checkFilter(mol)
        pains = drugfilters.checkPAINS(mol)
        
        mask.append(len(matchList) >= nFilters and pains is None)
        
    return df0[mask]

##############################
def calcSimilarity(sm1,sm2):
    # Input smiles strings, return similarity coefficient
    
    m1 = Chem.MolFromSmiles(sm1)
    m2 = Chem.MolFromSmiles(sm2)    

    fmgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2,atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())    
    ffp1 = fmgen.GetSparseCountFingerprint(m1)
    ffp2 = fmgen.GetSparseCountFingerprint(m2)
    similarity=DataStructs.DiceSimilarity(ffp1,ffp2)
    
    return similarity

##############################
def drawSampleLigs(ofname,df,N):
    # draw N ligands from the df, saves to png file  

    dfcut = df.head(N)

    pairs = [
        (Chem.MolFromSmiles(s),i)
        for s,i in zip(dfcut['smiles'],dfcut['ID'])
        if Chem.MolFromSmiles(s)
    ]
    mols,labels = zip(*pairs)
    img = Draw.MolsToGridImage(mols, molsPerRow= 6, subImgSize= (300,300), legends= labels) 
    
    #mols = [Chem.MolFromSmiles(smiles) for smiles in df['smiles'] if Chem.MolFromSmiles(smiles)]
    #labels = df['ID'].tolist()     
    #img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(300, 300),legends=labels)
    img.save(ofname)

    return 

##############################
def getProperties(df,exclude_nondruglike=False,nFilters=None):

    # Get smiles, chemical properties. dflist is a list of dfs
    # Excludes non druglike compounds (need to pass at least nFilters) if flag is True 
    
    # Add smiles strings to top hits
    df['smiles'] = df['posepath'].apply(lambda x: addSMILES(location_of_all_docks,x))
    ##########
    
    # Keep only drug-like compounds and append additional properties of interest
    props = df['smiles'].apply(compute_properties)
    df['MolWT'] = props.str[0]
    df['NumHeavyAtoms'] = props.str[1]
    df['LogP'] = props.str[2]        
    df['lid_root2'] = df['interface_delta_X'] / np.sqrt(df['MolWT'])
    
    # calculate Tanimoto similarity coefficient to compound list
    df['sim_rjw100'] = df['smiles'].apply(lambda x: calcSimilarity(rjw100, x))
    df['sim_6n10ca'] = df['smiles'].apply(lambda x: calcSimilarity(iuw, x))
    df['sim_4504'] = df['smiles'].apply(lambda x: calcSimilarity(smi_4504, x))
    df['sim_9647'] = df['smiles'].apply(lambda x: calcSimilarity(smi_9647, x))    
    
    ## Quick view compounds (caps to only showing ~30 compounds)
    #drawSampleLigs("temp_filtered_ligs.png",dflist[0],30)

    if exclude_nondruglike:
        df_druglike = excludeNonDrugLike(df,nFilters)
        return df_druglike
    
    return df

##############################
if __name__=="__main__": 
    main() 
