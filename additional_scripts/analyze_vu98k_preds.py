#!/usr/bin/env python3

'''
Analyze vu98k compounds: all, selected for testing, new putative hits. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import math
import seaborn as sbn
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from rdkit import Chem

##########

# this hit list is from Alexis, September 2025
hit_notflu=['VU0629007-1','VU0642620-1','VU0508002-1','VU0515209-1']
hit_fluo=['VU0528849-1','VU0621802-1','VU0614565-1']
nothit_fluo=['VU0629074-1', 'VU0646189-1', 'VU0628999-1', 'VU0542575-1', 'VU0614155-1']

hitlist=hit_notflu

plt.rcParams["font.size"] = 12

##############################
def main():
    
    vu98kDF = pd.read_csv("../out/list_all_vu98k.csv")
    vu98kDF_filtered = pd.read_csv("../out/1.5_list_filter_by_location_mark_confidence.csv")
    vu98kDF_tested = pd.read_csv("../out/vu98k_selected_for_testing.csv")

    vu98kDF = getCombined(vu98kDF) # all 98k
    vu98kDF_filtered = getCombined(vu98kDF_filtered) # 11k 
    vu98kDF_tested = getCombined(vu98kDF_tested) # 95
    vu98kDF_tested['hit_label'] = np.where(vu98kDF_tested['ID'].isin(hitlist), 1, 0)
    
    #########    
    # plots : preds v idx, enrichment 
    plotPreds(vu98kDF,vu98kDF_tested)
    enrichmentPlot_stepwise(vu98kDF_tested)
    #enrichmentPlot_stepwise(vu98kDF_filtered,step=250)

    ##########    
    # ranking w/o pred filters
    # what is the rank, %EF of the hit compounds based on idx?
    print("Out of all 98k, where do the hits rank?")
    vu98kDF = getRanking(vu98kDF)
    subset = vu98kDF[vu98kDF['ID'].isin(hitlist)][['ID', 'rank_idx','rank_sbp','rank_sap','rank_combo','rank_combo_idx_bp','rank_combo_idx_ap','rank_combo_bp_ap']]        
    print(subset)
    
    ##########
    # ranking , from confidence scores : result -> all hits are within 10% of vu98k
    print("Out of the 11k filtered list, where do the hits rank?")
    vu98kDF_filtered = getRanking(vu98kDF_filtered)
    subset = vu98kDF_filtered[vu98kDF_filtered['ID'].isin(hitlist)][['ID', 'rank_idx','rank_sbp','rank_sap','rank_combo','rank_combo_idx_bp','rank_combo_idx_ap','rank_combo_bp_ap']]                
    print(subset)

    ##########
    # are any of the tested compounds ranked within top 10% also of al vu98k?
    #print("Out of the 98k, where do the other 95 tested comopunds rank? Are there any ranked more highly than our hits?: SEE excel sheet cpd_rank_pct.csv" ) 
    #subset = vu98kDF[vu98kDF['ID'].isin(vu98kDF_tested['ID'])]
    ##[
    ##   ['ID', 'rank_idx_pct', 'rank_sbp_pct', 'rank_sap_pct', 'rank_combo_pct']    ]
    #subset.to_csv('cpd_rank_pct.csv',float_format='%.2f',index=False)

    ##########
    # was structural filtering necessary?
    # 95 tested hit v non hit energetics, can adjust for min_dist also 
    plot_box(vu98kDF_tested)

    # all vu98k , line plot
    plot_parallel_coords(vu98kDF,vu98kDF_tested) #_filtered)

    ##########
    # plot hit ranks
    plot_hit_ranks(vu98kDF,vu98kDF_filtered) # ranks plot

    ########## 
    ## was fatty acid mimetic substructure preference necessary ?                     
    vu98kDF_tested['fatty_acid_matches'] = vu98kDF_tested['smiles'].apply(
        lambda x: has_fatty_acid_mimetic(x, return_matches=True)
    )
    #vu98kDF_tested.to_csv('temp_substructure.csv',float_format='%.2f',index=False)
    check_substruct_enrich(vu98kDF_tested)

    ##########     
    # was docking to all 3 constructs necessary?
    check_multiple_constructs(vu98kDF_tested)
    
    return

##############################
def plotPreds(allDF,testedDF):

    # hit compounds
    hitsDF = testedDF[testedDF["ID"].astype(str).isin(hitlist)]

    preds = ['sigmoid_binder_pred','sigmoid_activity_pred']
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for i,pred in enumerate(preds):
        ax[i].scatter(allDF['interface_delta_X'].astype(float),allDF[pred].astype(float),color="gainsboro",alpha=0.5,s=8,label=f"VU98k")
        ax[i].scatter(testedDF['interface_delta_X'].astype(float),testedDF[pred].astype(float),color="cornflowerblue",s=20,label="Tested")
        ax[i].scatter(hitsDF['interface_delta_X'].astype(float),hitsDF[pred].astype(float),color="yellow", edgecolors="black",marker="*", s=130, label="Hit")        
        
        ax[i].set_ylim([0,1])        
        ax[i].set_xlabel('Interface score (REU)')
        ax[i].tick_params(axis="both", which="major")
        ax[i].legend(fontsize=11,loc='lower left')        
            
    ax[0].set_ylabel('Binder prediction')
    ax[1].set_ylabel('Activity prediction')

    plt.tight_layout()
    plt.savefig("plot_pred_v_idx.pdf", format="pdf")

    # 3d plot, for tested only
    figpred = plt.figure()
    axpred = figpred.add_subplot(111,projection='3d',computed_zorder=False)
    #figpred,axpred = plt.subplots(figsize=(4,3))
    axpred.scatter(testedDF['sigmoid_binder_pred'].astype(float),testedDF['sigmoid_activity_pred'].astype(float),testedDF['interface_delta_X'].astype(float),color="cornflowerblue",s=20,label="Tested compounds",depthshade=False)
    axpred.scatter(hitsDF['sigmoid_binder_pred'].astype(float),hitsDF['sigmoid_activity_pred'].astype(float),hitsDF['interface_delta_X'].astype(float),color="yellow", edgecolors="black",marker="*", s=130, label="Hit compounds",depthshade=False)        
    axpred.set_xlabel('Binder prediction')
    axpred.set_ylabel('Activity prediction')
    axpred.set_zlabel('Interface score (REU)')
    axpred.set_xlim([0.4,1])
    axpred.set_ylim([0.4,1])
    axpred.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_pred.png")
    
    return

##############################
def enrichmentPlot_stepwise(df,step=5):
    # intervals of 5, enrichment for hitlist from 95 tested (known fps)

    nmax = len(df)
    xvals = np.arange(step, nmax + 1, step)

    y_idx = enrichment_counts_tieavg(df,xvals, 'interface_delta_X', True)
    y_bp = enrichment_counts_tieavg(df,xvals, 'sigmoid_binder_pred', False)
    y_ap = enrichment_counts_tieavg(df,xvals, 'sigmoid_activity_pred', False)    
    y_combo = enrichment_counts_tieavg(df,xvals, 'combined_score', True)    

    fig,ax = plt.subplots(1,figsize=(7,3))
    ax.plot(xvals[:len(y_idx)], y_idx, label='Interface score',marker='s',markeredgecolor='k',c='peru') #darkturquoise')
    ax.plot(xvals[:len(y_bp)], y_bp, label='Binder prediction',marker='^',markeredgecolor='k',c='teal')
    ax.plot(xvals[:len(y_ap)], y_ap, label='Activity prediction',marker='H',markeredgecolor='k',c='limegreen')
    ax.plot(xvals[:len(y_combo)], y_combo, label='Combined',marker='o',markeredgecolor='k',c='maroon')

    y_random = [len(hitlist) * (n / len(df)) for n in xvals]
    ax.plot(xvals, y_random, label='Random', linestyle='--', c='gray', alpha=0.6)
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))    

    #ax.set_ylabel('Number of putative hits\n within set')
    ax.set_ylabel('Expected hits') # (with tie averaging)')
    ax.set_xlabel('Top N-ranked compounds')
    
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_enrich.png",dpi=300)
    
    return

##############################
def enrichment_counts_tieavg(df,xvals,sort_col,ascending=True):

    df_sorted = df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    out = []
    for n in xvals:
        if n > len(df_sorted):
            continue
        cutoff_score = df_sorted.iloc[n-1][sort_col]
        before = (
            df_sorted[sort_col] < cutoff_score
            if ascending
            else
            df_sorted[sort_col] > cutoff_score
        )
        tied = df_sorted[sort_col] == cutoff_score
        n_before = before.sum()
        remaining = n - n_before
        # contributions before tie
        hits_before = df_sorted.loc[before,'hit_label'].sum()
        # expected contribution from tied block
        tied_hit_fraction = df_sorted.loc[tied,'hit_label'].mean()
        expected_hits = (hits_before + remaining * tied_hit_fraction)
        out.append(expected_hits)            
    
    return out

##############################
def getRanking(df):
    
    df = df.copy()
    
    df['rank_idx'] = df['interface_delta_X'].rank(method='average', ascending=True)
    df['rank_sbp'] = df['sigmoid_binder_pred'].rank(method='average', ascending=False)
    df['rank_sap'] = df['sigmoid_activity_pred'].rank(method='average', ascending=False)
    df['rank_combo'] = df['combined_score'].rank(method='average', ascending=True)

    df['rank_combo_idx_bp'] = df['combined_score_idx_bp'].rank(method='average', ascending=True)
    df['rank_combo_idx_ap'] = df['combined_score_idx_ap'].rank(method='average', ascending=True)
    df['rank_combo_bp_ap'] = df['combined_score_bp_ap'].rank(method='average', ascending=True)    

    df['rank_idx_pct'] = (df['rank_idx']/len(df))*100
    df['rank_sbp_pct'] = (df['rank_sbp']/len(df))*100
    df['rank_sap_pct'] = (df['rank_sap']/len(df))*100
    df['rank_combo_pct'] = (df['rank_combo']/len(df))*100
    
    df['rank_combo_idx_bp_pct'] = (df['rank_combo_idx_bp']/len(df))*100
    df['rank_combo_idx_ap_pct'] = (df['rank_combo_idx_ap']/len(df))*100
    df['rank_combo_bp_ap_pct'] = (df['rank_combo_bp_ap']/len(df))*100        

    return df

##############################
def getCombined(df): 
    # combine the 3 metrics to get combined score 
    
    df = df.copy()
    df['combined_score'] = (
        df['interface_delta_X'].rank(method='average',ascending=True) +
        df['sigmoid_binder_pred'].rank(method='average',ascending=False) +
        df['sigmoid_activity_pred'].rank(method='average',ascending=False)
    )
    df['combined_score_idx_bp'] = (
        df['interface_delta_X'].rank(method='average',ascending=True) +
        df['sigmoid_binder_pred'].rank(method='average',ascending=False) )
    df['combined_score_idx_ap'] = (
        df['interface_delta_X'].rank(method='average',ascending=True) +
        df['sigmoid_activity_pred'].rank(method='average',ascending=False) )
    df['combined_score_bp_ap'] = (
        df['sigmoid_binder_pred'].rank(method='average',ascending=False) +
        df['sigmoid_activity_pred'].rank(method='average',ascending=False) )
    
    return df

##############################
def plot_box(df):

    term_names = ['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol',
                  'if_X_hbond_bb_sc','if_X_hbond_sc','interface_delta_X','total_score',
                  'sigmoid_binder_pred','sigmoid_activity_pred']
    #term_names = ['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol',
    #              'if_X_hbond_bb_sc','if_X_hbond_sc','interface_delta_X','total_score',
    #              'sigmoid_binder_pred','min_distance'] 
    
    nrows = 2
    ncols = math.ceil(len(term_names) / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6.5))
    axes = axes.flatten()

    non_hits = df[df['hit_label'] == 0]
    hit_df   = df[df['hit_label'] == 1].reset_index(drop=True)
    n_nothit = len(non_hits)
    n_hit    = len(hit_df)

    hit_markers = ['o', 's', '^', 'D']
    hit_colors  = ['firebrick', 'darkorange', 'purple', 'forestgreen']

    # x positions: boxplot at 0, hits spread to the right
    hit_xpos = [0.55 + j * 0.2 for j in range(n_hit)]  # 0.55, 0.75, 0.95, 1.15

    for i, term in enumerate(term_names):
        ax = axes[i]

        # boxplot for non-hits
        ax.boxplot(
            non_hits[term].dropna(),
            positions=[0],
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor='cornflowerblue', alpha=0.45, linewidth=1.2),
            medianprops=dict(color='navy', linewidth=1.8),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            showfliers=False
        )

        # jittered strip of non-hits
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(non_hits))
        ax.scatter(
            x=jitter,
            y=non_hits[term].values,
            color='cornflowerblue',
            alpha=0.4, s=18, zorder=2,
            edgecolors='none'
        )

        # hits side by side to the right of boxplot
        for j, (_, row) in enumerate(hit_df.iterrows()):
            ax.scatter(
                x=hit_xpos[j],
                y=row[term],
                color='yellow', #hit_colors[j],
                marker='*' , #hit_markers[j],
                s=90, zorder=5,
                edgecolors='black', linewidths=0.8,
                label=f'Hit {j+1}' if i == 0 else None
            )

        # vertical separator between boxplot and hits
        ax.axvline(x=0.35, color='gray', linewidth=0.7, linestyle=':', alpha=0.5)

        ax.set_xlim(-0.45, 1.35)
        ax.set_xticks([0] + hit_xpos)
        ax.set_xticklabels(
            [f'Non-hits\n(N={n_nothit})'] + [id[:-2] for id in hit_df['ID'].tolist()],
            fontsize=11, rotation=90
        )
        
        ax.set_ylabel(term, fontsize=11)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')

    # need to manualy force non decimal scales
    axes[4].yaxis.set_major_locator(MultipleLocator(2.0))
    axes[2].yaxis.set_major_locator(MultipleLocator(2.0))

    #### for min_dist plot 
    #axes[9].yaxis.set_major_locator(MultipleLocator(1.0))
    #axes[9].yaxis.set_minor_locator(MultipleLocator(0.5))    
    #axes[9].set_ylim(0,7)

    legend_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='cornflowerblue',
               markersize=7, alpha=0.7, label=f'Non-hit (N={n_nothit})')
    ]
    for j in range(n_hit):
        legend_handles.append(
            Line2D([0],[0], marker=hit_markers[j], color='w',
                   markerfacecolor=hit_colors[j], markeredgecolor='black',
                   markersize=8, label=f'Hit {j+1}')
        )
    axes[0].legend(handles=legend_handles, fontsize=9, framealpha=0.7)
    axes[0].legend().set_visible(False)

    plt.tight_layout()    
    plt.savefig("plot_95test_features.pdf", dpi=300)
    #plt.show()
    
    return

##############################
def plot_parallel_coords(df,df_tested):

    score_cols = ['interface_delta_X', 'sigmoid_binder_pred', 'sigmoid_activity_pred']
    axis_labels = ['Interface score\n(normalized, inverted)', 'Binder\nprediction', 'Activity\nprediction']

    df = df.copy()

    # normalize all scores to [0,1] where 1 = best
    # interface_delta_X: lower = better, so invert
    df['interface_norm'] = 1 - (df['interface_delta_X'] - df['interface_delta_X'].min()) / \
                               (df['interface_delta_X'].max() - df['interface_delta_X'].min())
    # binder and activity: higher = better
    df['binder_norm']   = (df['sigmoid_binder_pred'] - df['sigmoid_binder_pred'].min()) / \
                          (df['sigmoid_binder_pred'].max() - df['sigmoid_binder_pred'].min())
    df['activity_norm'] = (df['sigmoid_activity_pred'] - df['sigmoid_activity_pred'].min()) / \
                          (df['sigmoid_activity_pred'].max() - df['sigmoid_activity_pred'].min())

    norm_cols = ['interface_norm', 'binder_norm', 'activity_norm']

    non_hits = df[~df['ID'].isin(hitlist)]
    hit_df   = df[df['ID'].isin(hitlist)].reset_index(drop=True)

    hit_colors  = ['aqua','springgreen', 'orangered', 'fuchsia']
    hit_markers = ['o', 's', '^', 'D']

    # split non_hits :
    tested_ids = df_tested['ID'].tolist()
    tested_nonhits = df[df['ID'].isin(tested_ids) & ~df['ID'].isin(hitlist)]
    untested = df[~df['ID'].isin(tested_ids) & ~df['ID'].isin(hitlist)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.05, 1.1)

    x = [0, 1, 2]

    # untested compounds 
    for _, row in untested.iterrows():
        ax.plot(x, row[norm_cols].values,
                color='gainsboro', alpha=0.15, linewidth=0.6, zorder=1)
        
    # tested non-hits
    for _, row in tested_nonhits.iterrows():
        ax.plot(x, row[norm_cols].values,
                color='cornflowerblue', alpha=0.5, linewidth=0.6, zorder=2)
        
    # hits
    for j, (_, row) in enumerate(hit_df.iterrows()):
        ax.plot(x, row[norm_cols].values,
                color=hit_colors[j], linewidth=2.2, zorder=3,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.scatter(x, row[norm_cols].values,
                   color=hit_colors[j], marker=hit_markers[j],
                   s=50, zorder=4, edgecolors='black', linewidths=0.8)

    # axes
    for xi in x:
        ax.axvline(xi, color='gray', linewidth=1.2, alpha=0.5, zorder=2)
        ax.text(xi, -0.04, axis_labels[x.index(xi)],
                ha='center', va='top', fontsize=11,
                transform=ax.get_xaxis_transform())

    # y tick labels
    tick_labels = {0.0: '0.0\n(worst)', 0.2: '0.2', 0.4: '0.4',
                   0.6: '0.6', 0.8: '0.8', 1.0: '1.0\n(best)'}
    for val, lbl in tick_labels.items():
        ax.axhline(val, color='k', linewidth=0.5, linestyle=':', alpha=0.4, zorder=0)
        ax.text(-0.13, val, lbl, ha='right', va='center', fontsize=11, color='k')
    
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.spines[['top','bottom','left','right']].set_visible(False)
    ax.set_ylabel('Normalized score',  labelpad=30)

    # legend
    handles = [
        Line2D([0],[0], color='gainsboro',  alpha=1,linewidth=1.5,
               label=f'VU98k'),
        Line2D([0],[0], color='cornflowerblue', linewidth=1.5,
               label=f'Tested')] 
    
    for j, (_, row) in enumerate(hit_df.iterrows()):
        handles.append(
            Line2D([0],[0], color=hit_colors[j], linewidth=2,
                   marker=hit_markers[j], markerfacecolor=hit_colors[j],
                   markeredgecolor='black', markersize=7,
                   label=row['ID'].rsplit('-', 1)[0])
        )
    ax.legend(handles=handles, fontsize=10, framealpha=0.7,
              loc='lower left', bbox_to_anchor=(0,0))

    plt.tight_layout()
    plt.savefig("plot_parallel_coords.png", dpi=300)
    
    #plt.show()

##############################
def plot_hit_ranks(df,df_tested):

    rank_cols =['rank_idx','rank_sbp','rank_sap','rank_combo_idx_bp','rank_combo_idx_ap','rank_combo_bp_ap','rank_combo']
    #rank_labels =['Interface \n score','Binder \n prediction','Activity \n prediction','Combination','IDX+BP','IDX+AP','BP+AP']
    rank_labels =['IDX','BP','AP','IDX+BP','IDX+AP','BP+AP','Combination']            

    datasets = [
        (df,'98k full library'),
        (df_tested,'11k reduced set'),        
    ]

    # same color/markers as for parallel plot
    hit_colors  = ['aqua','springgreen', 'orangered', 'fuchsia']
    hit_markers = ['o', 's', '^', 'D']
    x = np.arange(len(rank_cols))

    fig, axes = plt.subplots(1,2,figsize=(8.5, 4),sharey=True)

    for ai, (idf,set_label) in enumerate(datasets):
        ax = axes[ai]
        hit_df = idf[idf['ID'].isin(hitlist)].reset_index(drop=True)

        # individual hit ranks
        for hi, (_,row) in enumerate(hit_df.iterrows()):
            ranks = row[rank_cols].values.astype(float)
            ax.scatter(x,ranks,color=hit_colors[hi],marker=hit_markers[hi],s=50,zorder=3,edgecolors='k',linewidths=1.5)
            
        ## max rank
        #max_ranks = hit_df[rank_cols].max().values.astype(float)
        #for ci, (max_rank,col) in enumerate(zip(max_ranks, rank_cols)):
        #    max_pct = (max_rank / len(idf)) *100
        #    ax.text(ci,max_rank*0.7,f'{max_rank:.0f}\n({max_pct:.0f}%)',fontsize=10,color='k',ha='center',va='top')
        
        #best_col_idx = np.argmin(max_ranks)
        #best_rank = int(max_ranks[best_col_idx])
        #best_pct = (best_rank/len(idf))*100
        #ax.text(best_col_idx,best_rank+20000, #*0.9,
        #        f'Rank={best_rank}\n({best_pct:.0f}%)',fontsize=10,color='k',ha='center',va='top')
        
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(rank_labels,rotation=90)
        ax.grid(True,alpha=0.4,which='both')        
        ax.set_ylabel('Rank (log scale)') # if ai == 0 else '') 
        #ax.set_xlim(-0.4,len(rank_cols) - 0.6

    # legend
    handles = []
    hit_df_any = df[df['ID'].isin(hitlist)].reset_index(drop=True)
    for hi, (_, row) in enumerate(hit_df_any.iterrows()):
        handles.append(
            Line2D([0],[0], marker=hit_markers[hi], color='w',
                   markerfacecolor=hit_colors[hi], markeredgecolor='black',
                   markersize=8, label=str(row['ID']).rsplit('-', 1)[0])
        )
    axes[1].legend(handles=handles, fontsize=11, framealpha=0.7, loc='upper left')
            

    plt.tight_layout()
    axes[1].legend().set_visible(False)
    axes[1].tick_params(labelleft=True)
    
    plt.savefig("plot_hit_ranks.pdf",dpi=300)    
        
    return 

##############################
def has_fatty_acid_mimetic(smiles, return_matches=False):
    # Return True if SMILES contains fatty-acid–like fragment
    # Optionally return matching pattern names

    fatty_acid_patterns = {
        "carboxyl": Chem.MolFromSmarts("C(=O)[OH]"),   # Carboxyl (-COOH)
        "ester": Chem.MolFromSmarts("C(=O)O"),         # Ester (-COOR)
        "amide": Chem.MolFromSmarts("C(=O)N"),         # Amide (-CONH2)
        "ether": Chem.MolFromSmarts("C-O-C"),          # Ether (-O-)
        "long_chain_C8+": Chem.MolFromSmarts("CCCCCCCC"),  # Long aliphatic chain (C8+)
        "chlorophenyl": Chem.MolFromSmarts("c1ccc(Cl)cc1")
    }

    if pd.isna(smiles):
        return [] if return_matches else False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [] if return_matches else False

    matches = [
        name for name, pat in fatty_acid_patterns.items()
        if mol.HasSubstructMatch(pat)
    ]

    if return_matches:
        return matches

    return len(matches) > 0

##############################
def check_substruct_enrich(df):
    motifs = ['carboxyl','ester','amide','ether','long_chain_C8+','chlorophenyl']
    
    for motif in motifs:
        hit_freq = (
            df[df['hit_label']==1]['fatty_acid_matches']
            .apply(lambda x: motif in x).mean()
        )

        nonhit_freq = (
            df[df['hit_label']==0]['fatty_acid_matches']
            .apply(lambda x: motif in x).mean()
        )

        print(motif, hit_freq, nonhit_freq)

##############################
def check_multiple_constructs(df):
    # in ~/Documents/lrh1/paper_screen98k/for_github/predict_vu98k_filter.py, I print out all predictions for all 3 constructs. that is the file loaded into allDF. 

    df = df.copy()
    
    allDF = pd.read_csv("temp_out_predict_vu98k_all_preds.csv")

    cols_to_show = ['ID', 'posepath', 'interface_delta_X', 'sigmoid_binder_pred','sigmoid_activity_pred']

    allDF_hits = allDF[allDF['ID'].isin(hitlist)]
    print(allDF_hits.sort_values('ID')[cols_to_show])

    allDF_hits.to_csv('temp_hits.csv',float_format='%.2f',index=False)

    ### ran the following in for_github, below is output min_dist: 
    #python 1_filter_by_location.py --input_file ../scripts/temp_hits.csv --output_file temp.csv --distance_cutoff 100 ; all are < 7A 
    # VU0629007-1 4.5766816
    # VU0642620-1 2.8614235
    # VU0508002-1 5.094739
    # VU0515209-1 3.3813248
    # VU0629007-1 2.5190148
    # VU0642620-1 2.6627507
    # VU0508002-1 3.347276
    # VU0515209-1 3.6410496
    # VU0642620-1 5.845037
    # VU0629007-1 2.5024621
    # VU0508002-1 2.4198058
    # VU0515209-1 5.5029635

    # what about non-hits, do they have a docked pose outside dist cutoff?
    #df_overlap = allDF[allDF['ID'].isin(df['ID'])]
    #df_overlap.to_csv('temp_tested.csv',float_format='%.2f',index=False)
    
    #i did: python 1_filter_by_location.py --input_file ../scripts/temp_tested.csv --output_file temp.csv --distance_cutoff 100 > ../scripts/temp_tested_mindist.csv
    # renamed to .ods to keep highlight/colors 
    # observations: 2 of the hits the top docked pose was for one construct, 2 of the hits it was for another. all 3 constructs of the 4 hits docked met distance cutoffs. 14 out of the 95 tested compounds had at least one pose to one of the constructs that did not meet distance cutoff criteria (could be used to deprioritize?). looking at the docked poses, they are all over the place

    ## plot distances for the constructs 
    df_mindist = pd.read_csv("temp_tested_mindist.csv",header=None)
    df_mindist.columns = ['ID', 'min_dist','pdbpath']
    df_mindist["state"] = df_mindist["pdbpath"].apply(get_state)

    # order: hits first, then rest
    #ordered_ids = (
    #    list(df_mindist[df_mindist['ID'].isin(hitlist)]['ID'].unique()) +
    #    list(df_mindist[~df_mindist['ID'].isin(hitlist)]['ID'].unique())
    #)
    # force order to match with the other min distance plot
    hit_ids = [i for i in hitlist if i in df_mindist['ID'].values]
    other_ids = [i for i in df_mindist['ID'].unique() if i not in hit_ids]
    ordered_ids = hit_ids + other_ids
    
    # remove duplicates while preserving order
    seen = set()
    ordered_ids = [x for x in ordered_ids if not (x in seen or seen.add(x))]
    
    # map IDs to x positions
    id_to_x = {id_: i for i, id_ in enumerate(ordered_ids)}
    
    fig, ax = plt.subplots(figsize=(10, 3))

    color_map = {
        "iuw": "khaki",
        "iuw_4pld": "salmon",
        "iuw_7xnk": "darkcyan"
        #"unknown": "gray"
    }

    df_mindist["x"] = df_mindist["ID"].map(id_to_x)
    df_mindist = df_mindist.dropna(subset=["x"])

    df["state"] = df["posepath"].apply(get_state)
    highlight_map = dict(zip(df["ID"], df["state"]))
    df_mindist["highlight"] = df_mindist.apply(
        lambda r: r["state"] == highlight_map.get(r["ID"], None),
        axis=1
    )
    bg = df_mindist[~df_mindist["highlight"]]
    hi = df_mindist[df_mindist["highlight"]]

    # background poses (faded)
    ax.scatter(
        bg["x"],
        bg["min_dist"],
        c=bg["state"].map(color_map),
        edgecolor="k",
        linewidth=0.5,
        alpha=0.7,
        s=35,
        label="other poses"
    )
    
    # highlighted poses (strong)
    ax.scatter(
        hi["x"],
        hi["min_dist"],
        c=hi["state"].map(color_map),
        edgecolor="k",
        alpha=1.0,
        s=35,
        linewidth=1.2,
        label="selected pose"
    )

    ax.set_xticks(range(len(ordered_ids)))
    ax.set_xticklabels(
        [str(x).split('-')[0] for x in ordered_ids],
        fontsize=7,
        rotation=90
    )

    ax.set_xlim(-0.5, len(ordered_ids) - 0.5)
    ax.set_ylim(1,10.5)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    # dist threshold
    ax.axhline(7.0, color='red', linestyle='--', linewidth=1,alpha=0.7)

    ax.set_ylabel('Minimum distance')
    ax.grid(True,alpha=0.4,which='both')        
    
    plt.tight_layout()
    #plt.show()
    plt.legend()

    legend_handles = [
        Line2D(
            [],
            [],
            color=color_map[state],
            marker='o',
            linestyle='None',
            markersize=7,
            label=state,
            markeredgecolor='k',
            markeredgewidth=0.5,
            alpha=0.9
        )
        for state in color_map.keys()
    ]
    highlight_handle = Line2D(
        [],
        [],
        color="black",
        marker='o',
        linestyle='None',
        markersize=7,
        markerfacecolor="none",
        label="Selected pose",
        markeredgewidth=1.5,
    )
    ax.legend(handles=legend_handles + [highlight_handle])
    
    plt.savefig('plot_mindist_95_allconstructs.pdf',dpi=300)

    return 

##############################
def get_state(s):
    s = str(s)
    if "iuw_4pld" in s:
        return "iuw_4pld"
    elif "iuw_7xnk" in s:
        return "iuw_7xnk"
    elif "iuw" in s:
        return "iuw"
    else:
        return "unknown"


##############################
if __name__=="__main__":

    main()
