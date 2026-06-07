#!/usr/bin/env python3

# Plot heat map of residue contacts

import os
import sys
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.size"] = 12

activators_list='/path/to/list/of/compounds/that/are/regulators.csv'

# aa dictionary
aa = {
    'ALA':'A', 
    'CYS':'C', 
    'ASP':'D', 
    'GLU':'E', 
    'PHE':'F', 
    'GLY':'G', 
    'HIS':'H', 
    'ILE':'I', 
    'LYS':'K', 
    'LEU':'L', 
    'MET':'M', 
    'ASN':'N', 
    'PRO':'P', 
    'GLN':'Q', 
    'ARG':'R', 
    'SER':'S', 
    'THR':'T', 
    'VAL':'V', 
    'TRP':'W', 
    'TYR':'Y' 
}

def main(): 

    # List of file names
    pwd='.'
    file_names = glob.glob(os.path.join(pwd, "./out_res_rl/*_res.dat"))

    # initialize empty dataframe
    active = pd.read_csv(activators_list)
    active_vuid = set(active['VU Number'].astype(str).map(clean_id))

    data_list = []

    # Read data from each file and append to combinedDF
    for ifname in file_names:
        if 'VU' not in ifname:
            continue

        # find vuid
        match = re.search(r'VU(\d+)',ifname)
        if not match:
            continue
        vuid = match.group(0)

        # read data
        data = pd.read_csv(ifname, sep='\s+', skiprows=1, names=["#Res1", "#Res2", "TotalFrac", "Contacts"])
        data["vuid"] = vuid        
        data_list.append(data)

    # concatenate
    if data_list:
        combinedDF = pd.concat(data_list,ignore_index=True)
    else:
        sys.exit("error, no files read")

    # pivot data
    pivotDF = (
        combinedDF
        .pivot_table(index="vuid", columns="#Res2", values="TotalFrac",aggfunc="sum", fill_value=0)
        .astype(float)
        .rename_axis(None,axis=1)
        #.reset_index()
    )

    pivotDF.index = pivotDF.index.map(clean_id)

    # sanity check
    if pivotDF.index.duplicated().sum() > 0:
        print("Warning: duplicate vuid detected")
        pivotDF = pivotDF[~pivotDF.index.duplicated(keep='first')]
  
    # update residue numbering in pivotDF to match canonical (rather than sequential) numbering
    renumDF = pd.read_csv('renum.txt', sep='\s+')
    mapping = dict(zip(renumDF['renum'],renumDF['orig']))
    pivotDF = pivotDF.rename(columns=mapping)
        
    ## separate activator only df  , copy
    activeDF = pivotDF.loc[pivotDF.index.isin(active_vuid)].copy()

    # frequency of residues 
    all_freq = ( (pivotDF > 0).sum(axis=0) )  / pivotDF.shape[0] # normalized
    active_freq = ( (activeDF > 0).sum(axis=0) ) / activeDF.shape[0] # normalized

    fig,ax= plt.subplots(1,figsize=(7,2.5))

    ax.scatter(active_freq.index, active_freq.values,c='yellow',marker='o',edgecolor='k',alpha=0.8,label=f'Activators (N={activeDF.shape[0]})')    
    ax.scatter(all_freq.index, all_freq.values,c='cornflowerblue',marker='o',edgecolor='k',alpha=0.6,label=f'S2k (N={pivotDF.shape[0]})')
    
    ax.set_ylim([-0.05,1.05])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(1))    
    ax.grid(True, alpha=0.3, axis='both')

    ax.set_xlabel("Residue")
    ax.set_ylabel("Normalized frequency")
    plt.legend(fontsize=11)
    plt.tight_layout()  
    #plt.show()
    plt.savefig("contacts.pdf",dpi=300)

    return 

##############################
def clean_id(x):
    return str(x).strip().split('-')[0]

##############################

if __name__ == "__main__":
    main()
