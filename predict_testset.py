#!/usr/bin/env python3

'''
predict_testset.py: Generate predictions from any compound library. 
Need CSV file with RosettaLigand energy terms from docked pose. 
Minimal code needed to generate predictions.

To execute: python predict_testset.py <input CSV file> [output CSV file]

Example data for input CSV file (copy from input_dummy_eterms.csv): 

name,if_X_fa_atr,if_X_fa_elec,if_X_fa_rep,if_X_fa_sol,if_X_hbond_bb_sc,if_X_hbond_sc
cpd1,-23.446,0.171,1.277,1.832,0.000,-0.073
cpd2,-24.506,0.142,2.628,1.763,0.000,0.000
cpd3,-30.935,-0.967,2.891,8.658,-2.204,-2.709
cpd4,-35.391,-1.057,3.364,11.697,-3.813,-4.228
'''

import sys
import pandas as pd 
import argparse
import torch
import numpy as np

# ANN architecture and data format 
import utils_drugfilters as drugfilters
from utils_define_data_model import Network,NetworkActivity
import predict

##############################
def main(ifname,ofname):

    eterms_list=['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol','if_X_hbond_bb_sc','if_X_hbond_sc']
    
    model_blipl = 'blip-l.pth'
    model_alipl = 'alip-l.pth'

    df=pd.read_csv(ifname)

    # program exits if all eterms are not present in inputCSV 
    all_eterms_present = set(eterms_list).issubset(df.columns)
    if not all_eterms_present:
        sys.exit("Error: Required energy term columns are not all listed in inputCSV. See eterms_list for required eterms.")

    # predict and append predictions to df

    for index,row in df.iterrows():
        features = row[eterms_list].to_numpy()
        with torch.inference_mode():
            preds,_ = predict.runTestSet(Network,model_blipl,[features],1)
            df.at[index, 'binder_pred'] = preds.item()
                                                       
    eterms1=eterms_list.copy()
    eterms1.append('binder_pred')            
    for index,row in df.iterrows():
        features = row[eterms1].to_numpy()
        with torch.inference_mode():        
            preds,_ = predict.runTestSet(NetworkActivity,model_alipl,[features],1)
        df.at[index, 'activity_pred'] = preds.item()

    # calculate sigmoid of preds 
    df['sigmoid_binder_pred'] = 1 / (1 + np.exp(-df['binder_pred']))
    df['sigmoid_activity_pred'] = 1 / (1 + np.exp(-df['activity_pred']))

    # write df with predictions to file
    df.to_csv(ofname,index=False,float_format="%.2f") 
        
    return 
    
##############################
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("inCSV",help="Input CSV file. See predict_testset.py header for sample file.")
    parser.add_argument("outCSV",nargs="?",default="out_predict_testset.csv", help="Output CSV file.")
    
    args = parser.parse_args()
    if args.outCSV == "out_predict_testset.csv": 
        print("Warning: No output file specified. Using default: out_predict_testset.csv")
        
    if len(sys.argv) < 2:
        print("USAGE: python predict_testset.py <input.csv> [output.csv]")
        sys.exit(1)
        
    main(args.inCSV,args.outCSV) 
        
