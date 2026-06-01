#!/usr/bin/env python3

'''
1.5_mark_confidence.py:

Reads CSV with compounds list that contains energy terms, predictions, chemical properties, SMILES.

Appends "confidence" column based on metric cutoffs (defined within the function).

Usage: python 1.5_mark_confidence.py --input_file inlist.csv --output_file outlist.csv
'''

import argparse
import pandas as pd
import numpy as np

##############################
def main(ifname,ofname):
    
    df = pd.read_csv(ifname)
    
    # sanity check
    if 'lid_root2' not in df or 'binder_pred' not in df or 'activity_pred' not in df:
        print(f"Skipping {ifname.name}. lid_root2 or predictions missing.")

    conditions = [
        (df['sigmoid_binder_pred'] > 0.8) & (df['sigmoid_activity_pred'] > 0.7),
        ((df['sigmoid_binder_pred'] > 0.8) | (df['sigmoid_activity_pred'] > 0.7)) & 
        (df['sigmoid_binder_pred'] > 0.5) & (df['sigmoid_activity_pred'] > 0.5)
    ]
    choices = ['high', 'medium']
    
    df['confidence'] = np.select(conditions, choices, default='low')

    order=['high','medium','low']
    df['confidence'] = pd.Categorical(df['confidence'],categories=order,ordered=True)
    df = (df.sort_values('confidence').drop_duplicates(subset='ID',keep='first'))
    
    df.to_csv(ofname,index=False)

    return 

##############################
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Labels compounds as high, medium, low confidnece based on binder and activity predictions, saves output to csv file.")

    parser.add_argument("--input_file", required=True, help="Input CSV files with compounds.")
    parser.add_argument("--output_file", required=True, help="Output CSV containing compounds with confidence scores.")
    args = parser.parse_args()
    
    main(args.input_file,args.output_file)
