#!/usr/bin/env python3

'''
Get similarity of df compounds per pair (cpd-cpd)
Create heatmap of similairities. use for tested compounds
Use same fingerprint defintion as in calc_tanimoto.py
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sbn
from itertools import combinations

from rdkit import Chem, DataStructs
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem import rdFingerprintGenerator

##############################
fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,   
    useBondTypes=False       # ignore bond order 
)
plt.rcParams["font.size"] = 12
######################################################################
def main():

    df = pd.read_csv('../out/vu98k_selected_for_testing.csv')

    fps, labels = smiles_to_fps(df, smiles_col='smiles', label_col='ID')
    print(f"{len(fps)} / {len(df)} compounds successfully parsed")

    ncpds = len(fps)
    
    tnmat = np.ones((ncpds,ncpds))
    for i,j in combinations(range(ncpds),2):
        score = DataStructs.TanimotoSimilarity(fps[i],fps[j])
        tnmat[i,j] = score
        tnmat[j,i] = score

    tnDF  = pd.DataFrame(tnmat,index=labels,columns=labels)

    # heatmap
    fig,ax = plt.subplots(figsize=(14,14))
    mask = np.tril(np.ones_like(tnmat,dtype=bool))
    sbn.heatmap(
        tnDF,
        #mask=mask,
        ax=ax,
        cmap='magma',
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Tanimoto similarity','shrink':0.8}
    )

    ax.tick_params(axis='x',labelsize=7,rotation=90)
    ax.tick_params(axis='y',labelsize=7,rotation=0)

    plt.tight_layout()
    plt.savefig('plot_tnheatmap.pdf',dpi=300)

    return 

##############################
def smiles_to_fps(df, smiles_col, label_col):
    labels = []
    fps = []
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(row[smiles_col])
        if m:
            m = Chem.RemoveHs(m)
            fps.append(fpgen.GetFingerprint(m))
            labels.append(row[label_col])   # only added if mol is valid
        else:
            print(f"WARNING: could not parse SMILES for {row[label_col]}, skipping")
    return fps, labels


########################################
if __name__=="__main__": 
    main() 

