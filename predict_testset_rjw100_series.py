#!/usr/bin/env python3

''' 
predict_testset_rjw100.py : uses saved models BLiP-L and ALiP-L to predict binder,activity for RJW100-series compounds.
Includes predicting from ligands with published xtal structures, ligands from congeneric series. 

Usage: python predict_testset_rjw100_series.py 
'''

import sys
import numpy as np 
import pandas as pd
import torch 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

# ANN architecture and data format 
from utils_define_data_model import Network,NetworkActivity
import predict

plt.rcParams["font.size"] = 12

##############################
def main():

    eterms_list=['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol','if_X_hbond_bb_sc','if_X_hbond_sc']
    
    model_blipl = 'blip-l.pth'
    model_alipl = 'alip-l.pth'
    
    # Testing on out-of-distribution ligands. The predHoldOutSet functions contain prediction, labeling, plotting scripts.
    # Input energy terms, in form of CSV, defined inside each function.
    
    ## Ligands bound to LRH-1 in published xtals
    ifname="../data/xtal_rjw100_series/out_eterms_xtal.csv"
    refmap=["../data/xtal_rjw100_series/master_list.csv","pdb",ifname,"description"]
    xtalDF = predHoldOutSet("xtal",model_blipl,model_alipl,eterms_list,ifname,refmap)
    
    ## Ligands bound to LRH-1 from congeneric series design ligands
    ifname="../data/xtal_rjw100_series/out_eterms_design.csv"
    refmap=["../data/xtal_rjw100_series/master_design.csv","ligand",ifname,"description"]
    designDF = predHoldOutSet("design",model_blipl,model_alipl,eterms_list,ifname,refmap)
    
    ## Analysis, plots for visualization
    plotListMetrics(xtalDF,designDF)
    plotExpvMetrics(designDF,xtalDF) 
    
    return

##############################
def predHoldOutSet(setName,modelBinder,modelActivity,eterms,inputCSV,refmap=None):
    '''
    Use modelBinder and modelActivity to predict binder, activity for the inputs in inputCSV.
    setName: user-defined name for this set; will be used later if sets are compared with each other
    eterms: Energy terms needed for predictions, inputCSV should have these listed per entry
    refmap: if set, mapping to match entries beteen inputCSV and a reference CSV
    refmap format is [master list CSV, column in master list to match with inputCSV, inputCSV, column in inputCSV to match with master ]
    '''

    df=pd.read_csv(inputCSV)

    # program exits if all eterms are not present in inputCSV 
    all_eterms_present = set(eterms).issubset(df.columns)
    if not all_eterms_present:
        sys.exit("Error: Required energy term columns are not all listed in inputCSV. See eterms_list for required eterms.")

    # predict and append predictions to df

    for index,row in df.iterrows():
        features = row[eterms].to_numpy()
        with torch.inference_mode():
            preds,_ = predict.runTestSet(Network,modelBinder,[features],1)
        df.at[index, 'binder_pred'] = preds.item()

    eterms1=eterms.copy()
    eterms1.append('binder_pred')            
    for index,row in df.iterrows():
        features = row[eterms1].to_numpy()
        with torch.inference_mode():        
            preds,_ = predict.runTestSet(NetworkActivity,modelActivity,[features],1)
        df.at[index, 'activity_pred'] = preds.item()
    
    # if refmap defined, append refmap labels to df
    assert(refmap[2]==inputCSV)
    refDF = pd.read_csv(refmap[0])
    outDF = df.merge(
        refDF,
        left_on=refmap[3],
        right_on=refmap[1],
        how='left',
        validate='many_to_one' # errors on duplicates 
    )
    
    outDF['set_name'] = setName 

    return outDF

##############################
def plotListMetrics(xtalDF,designDF):
    # Plot idx, preds from each df  by the metric. X axis is ligand. 

    # combine dataframes
    comboDF = pd.concat([xtalDF,designDF])
    comboDF['sigmoid_binder_pred'] = 1 / (1 + np.exp(-comboDF['binder_pred']))
    comboDF['sigmoid_activity_pred'] = 1 / (1 + np.exp(-comboDF['activity_pred']))

    # marker styles based on source
    fig,ax=plt.subplots(3,1,figsize=(7,8))    
    marker_styles = {'xtal': 'p', 'design': 'o'}
    colors = {'xtal': 'tab:blue', 'design': 'tab:red'}
    sizes = {'xtal': 60, 'design': 40}    
    labels = {'xtal': 'NR5A2 crystal structure with bound compound', 'design': r"$\it{In\ silico}$ modification to positive control RJW100"}    
    for label in comboDF['set_name'].unique():
        subset = comboDF[comboDF['set_name'] == label]
        facecolors = ['white' if v == 'ia' else colors[label] for v in subset['ec50_avg']]  # denote inactives
        ax[0].scatter(subset['ligand'], subset['interface_delta_X'],
                      marker=marker_styles[label], facecolors=facecolors, s=sizes[label],
                      edgecolor='k',
                      label=labels[label])
        ax[1].scatter(subset['ligand'], subset['sigmoid_binder_pred'],
                      marker=marker_styles[label], facecolors=facecolors, s=sizes[label],
                      edgecolor='k',
                      label=labels[label])
        ax[2].scatter(subset['ligand'], subset['sigmoid_activity_pred'],
                      marker=marker_styles[label], facecolors=facecolors, s=sizes[label],
                      edgecolor='k',
                      label=labels[label])

    ax[0].set_ylabel('IDX (REU)',fontsize=12)
    ax[1].set_ylabel('Binder Prediction',fontsize=12)
    ax[2].set_ylabel('Activity Prediction',fontsize=12)
    ax[0].set_xticklabels([]),    ax[1].set_xticklabels([])
    [ax[i].tick_params(axis="both", which="major", labelsize=12) for i in range(3)]

    # for idx plot 
    ax[0].yaxis.set_major_locator(MultipleLocator(2))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_ylim([-30.5,-16])
    
    for i in [1,2]: # binder and activity pred plots
        ax[i].yaxis.set_major_locator(MultipleLocator(0.1))
        ax[i].yaxis.set_minor_locator(MultipleLocator(0.05))
        ax[i].set_ylim([0,1.04])
        
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.savefig('result_testset_rjw100_series.pdf',dpi=300)            
    
    return

##############################
def plotExpvMetrics(df0,refdf0):
    # analouges plot preds vs ec50 , re_rjw100

    df = df0.copy()
    refdf = refdf0.copy()
    
    # fill non numberic entries with "-1"
    cols = ["ec50_avg","ec50_std","re_rjw100"]
    df[cols] = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    df['sigmoid_binder_pred'] = 1 / (1 + np.exp(-df['binder_pred']))
    df['sigmoid_activity_pred'] = 1 / (1 + np.exp(-df['activity_pred']))
    refdf['sigmoid_binder_pred'] = 1 / (1 + np.exp(-refdf['binder_pred']))
    refdf['sigmoid_activity_pred'] = 1 / (1 + np.exp(-refdf['activity_pred']))
    
    figexp,axexp=plt.subplots(3,2,figsize=(7,8))
    x_cols = ['ec50_avg', 're_rjw100']
    y_cols = ['interface_delta_X', 'sigmoid_binder_pred', 'sigmoid_activity_pred']
    
    for i, y in enumerate(y_cols):
        for j, x in enumerate(x_cols):
            ax = axexp[i][j]

            # masks
            mask_act = df[x] != -1
            mask_ia = df[x] == -1             
            
            # df without inactives
            ax.scatter(
                df.loc[mask_act,x],
                df.loc[mask_act,y],
                color='cornflowerblue',
                edgecolor='k',
                s=30,
                label="Actives"
            )

            # df with inactives for binding only           
            if x == "ec50_avg" and mask_ia.any():
                x_ia_plot = df.loc[mask_ia, x].copy()
                x_ia_plot[:] = df.loc[mask_act, x].max() + 1
            else:
                x_ia_plot = df.loc[mask_ia, x]

            #if x == "ec50_avg": # don't plot inactives for RE 
            ax.scatter(
                x_ia_plot,
                df.loc[mask_ia,y],
                color='lavender',
                edgecolor='k',
                s=30,
                label="Inactives"
            )

            #if x == "ec50_avg": 
            rho,pval = spearmanr(df[x],df[y])
            #else: # for RE, exclude inactives from correlation 
            #    rho,pval = spearmanr(df.loc[mask_act,x],df.loc[mask_act,y])
                
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = ""
            
            axexp[i][j].text(
                0.63, 0.15,
                # f"ρ = {rho:.2f}\np = {pval:.2e}",
                f"ρ = {rho:.1f}{sig}",
                transform=axexp[i][j].transAxes,
                verticalalignment='top'
            )
            
            # plot ec50 and re values for reference pdb with rjw100
            # use same marker plot style from xtal plotting above 
            # ec50,re values are from mays 2019 j med chem
            val = refdf.loc[refdf["pdb"] == "5l11", y].values[0]
            if x == "ec50_avg" : # EC50=1.5 +- 0.4 uM
                ax.scatter(1.5,val,color='deeppink',edgecolor='k',s=60,marker='p',label="RJW100")
            elif x == "re_rjw100": # RE = 1.0
                ax.scatter(1.0,val,color='deeppink',edgecolor='k',s=60,marker='p',label="RJW100")
            
    # formatting , same as other plot
    
    for j in [0,1]:     
        for i in [1,2]: # binder and activity pred plots
            axexp[i][j].yaxis.set_major_locator(MultipleLocator(0.1))
            axexp[i][j].yaxis.set_minor_locator(MultipleLocator(0.05))
            axexp[i][j].set_ylim([0,1.04])
        axexp[0][j].yaxis.set_major_locator(MultipleLocator(2))
        axexp[0][j].yaxis.set_minor_locator(MultipleLocator(1))
        axexp[0][j].set_ylim([-30.5,-16])

    axexp[2][0].set_xlabel(r'EC$_{50}$ ($\mu$M)')
    axexp[2][1].set_xlabel('RE')

    axexp[0][0].set_ylabel('IDX (REU)')
    axexp[1][0].set_ylabel('Binder prediction')
    axexp[2][0].set_ylabel('Activity prediction')
            
    plt.legend(borderpad=0.3,labelspacing=0.3,loc='lower right')
    plt.savefig('plot_corr.pdf',dpi=300)            

    return 

########################################
if __name__=="__main__": 
    main() 
