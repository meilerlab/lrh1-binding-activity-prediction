#!/usr/bin/env python3

'''
get_shortlist.py : 

Reads CSV with compounds list that contains energy terms, predictions, chemical properties, SMILES. 

Prints a "short" list that was used to then visually inspect docked poses.  

Usage: python 2_get_shortlist.py <in csvs> <column to sort by> <rank cutoff> <output file name> --mode <mode options see below> --exclude <optional, compounds to exclude from ranking list> (if applicable) --descend 

'''

import argparse
import pandas as pd
from pathlib import Path
from rdkit import Chem

##############################

def main(inCSVs,sortby,topN,mode,ascending,ofname,exclude_ids=None):

    shortlist=[]
    
    for ifname in Path().glob(inCSVs):
        df = pd.read_csv(ifname)

        # sanity check
        if 'lid_root2' not in df or 'smiles' not in df:
            print(f"Skipping {ifname.name}. lid_root2 or smiles missing.")

        print(f"Num. unique entries for {sortby}: {df[sortby].nunique()}")

        # optional exclude
        if exclude_ids:
            df = df[~df['ID'].astype(str).isin(exclude_ids)]
            
        ##########

        ## USER can specify mode in command line 

        if mode == "head_tail_pattern":
            # 0. list compounds that contain head-tail pattern 
            temp_shortlist = df[df['smiles'].apply(has_head_tail_pattern)]
        elif mode == "fatty_acid_mimetic_search":
            # 1. list compounds that contain fatty acid mimetic fragments
            temp_shortlist = df[df['smiles'].apply(has_fatty_acid_mimetic)]
        elif mode == "no_substructure_search"  :
            # 2. Do not use a substructure search 
            temp_shortlist = df
        
        ##########

        shortlist.append(temp_shortlist)

    # combine all frag hits lists
    df_shortlist = pd.concat(shortlist,ignore_index=True)
    
    df_shortlist_unique = (
        df_shortlist.sort_values(sortby).drop_duplicates(subset=["ID"], keep="first")    )
    
    # get top N ranked 
    if topN == -1 : # don't rank
        topdf=df_shortlist_unique.copy()
    else: 
        df_shortlist_unique["rank"] = df_shortlist_unique[sortby].rank(method="dense", ascending=ascending)
        topdf = df_shortlist_unique[df_shortlist_unique["rank"] <= topN].copy()

    topdf.to_csv(ofname, index=False)
    print(f"\nWrote {ofname} with shortlist, unique entries for molecules.")

    ##########    
    # To visualize poses in pymol: include path to docked poses in the function. Prints in chunks (N). 
    printPoses_forpymol("temp_out_pymol_lists.txt",topdf,100)
    
    return

##############################
def has_fatty_acid_mimetic(smiles):
    # Return True if SMILES contains fatty-acid–like fragment

    fatty_acid_patterns = [
        Chem.MolFromSmarts("C(=O)[OH]"),  # Carboxyl (-COOH) 
        Chem.MolFromSmarts("C(=O)O"),     # Ester (-COOR)
        Chem.MolFromSmarts("C(=O)N"),     # Amide (-CONH2)
        Chem.MolFromSmarts("C-O-C"),      # Ether (-O-)
        Chem.MolFromSmarts("CCCCCCCC"),   # Long aliphatic chain (C8+)
        Chem.MolFromSmarts("c1ccc(Cl)cc1") 
    ]
    
    if pd.isna(smiles):
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    return any(mol.HasSubstructMatch(pat) for pat in fatty_acid_patterns)

##############################
def has_head_tail_pattern(smiles):
    # Returns True if molecule has both an acidic head and a hydrophobic tail
    
    acidic_head_patterns = [
        Chem.MolFromSmarts("C(=O)[OH]"),  # Carboxyl (-COOH) 
        Chem.MolFromSmarts("S(=O)(=O)[O]"),  # Sulfonic acid (-SO3H)
        Chem.MolFromSmarts("P(=O)(O)(O)")  # Phosphonic acid (-PO3H2)
    ]
    
    hydrophobic_tail_patterns = [
        Chem.MolFromSmarts("CCCCCCCC"),  # Long aliphatic chain (C8+)
        Chem.MolFromSmarts("[a]")  # Aromatic ring
    ]
    
    if pd.isna(smiles):
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    has_head = any(
        mol.HasSubstructMatch(pat)
        for pat in acidic_head_patterns
    )

    if not has_head:
        return False

    has_tail = any(
        mol.HasSubstructMatch(pat)
        for pat in hydrophobic_tail_patterns
    )

    return has_tail

##############################
def printPoses_forpymol(ofname,df,nchunks=10):
    # just prints "pymol pose1.pdb pose2.pdb .. etc for quick viewing in pymol
    
    location_of_all_docks='~/ulls/vulibrary/' # location of all docked poses
    
    poses = df["posepath"]
    
    #for i in range(0, len(poses), nchunks):
    #    chunk = poses.iloc[i:i+nchunks]
    #    print("pymol " + " ".join(location_of_all_docks + p for p in chunk))
    #    print("\n")

    with open(ofname, "w") as f:
        for i in range(0, len(poses), nchunks):
            chunk = poses.iloc[i:i+nchunks]
            f.write("pymol " + " ".join(location_of_all_docks + p for p in chunk) + "\n\n")
            
    print(f"\nWrote {ofname}. Delete this file if not needed.\n")
            
    return 
        
##############################
if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Process CSV files, sort by a column, take top N, optional exclude folder, optional descending sort"
    )
    
    # positional arguments
    parser.add_argument("ifname", type=str, help="Input CSV files with compounds from which to select")
    parser.add_argument("sortby", type=str, help="Column to sort by")
    parser.add_argument("topN", type=int, help="Number of top rows to keep (-1 for all)")
    parser.add_argument("ofname", type=str, help="Output CSV containig filtered list")
    parser.add_argument("--mode", required=True,choices=["no_substructure_search","head_tail_pattern","fatty_acid_mimetic_search"],help="Select execution mode")    
    
    # optional flags
    parser.add_argument("--exclude", type=Path, default=None, help="Folder with CSVs containing IDs to exclude")
    parser.add_argument("--descend", action="store_true", help="Sort in descending order (default: ascending)")

    # save to variables
    args = parser.parse_args()
    ifname = args.ifname
    sortby = args.sortby
    topN = args.topN
    ofname = args.ofname
    excludeLists = args.exclude
    ascending = not args.descend
    mode = args.mode

    # add option to exclude compounds in already checked lists
    excludeLists = Path(excludeLists) if excludeLists is not None else None
    if excludeLists and excludeLists.exists():
        exclude_ids = set()
        for exfile in excludeLists.glob("*.csv"):
            df_exclude = pd.read_csv(exfile)
            if 'ID' in df_exclude:
                exclude_ids.update(df_exclude['ID'].astype(str))
        print(f"Marked {len(exclude_ids)} unique IDs from exclude folder.")
    else:
        exclude_ids = None
        print("\nNo files found to exclude.")
        
    
    main(ifname,sortby,int(topN),mode,ascending,ofname,exclude_ids)
