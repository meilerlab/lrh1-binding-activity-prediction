#!/usr/bin/env python3

''' 
train_crossval_save.py : code to train and perform cross-validation of BLiP-L and ALiP-L. 
Trained models are saved separately to .pth files. 

Usage: python train_crossval_save.py --training_mode <crossval,save> --seed <hard set seed> --jumbled 
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

# ANN architecture and data format 
from utils_define_data_model import BioData,Network,NetworkActivity
import predict

##############################
# global variables 
nExpCut = 0.45 # for individual experiment
SEED = 42 # random_state seed used for final model training 
RUN_JUMBLED = False # to run permutation test with feature jumbling 
blipl_params = [1e-4, 3000] # lr, maxEpochs
alipl_params = [1e-5, 3000] # lr, maxEpochs
torch.set_printoptions(precision=6, sci_mode=False)
plt.rcParams["font.size"] = 12
############################################################
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0): 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean',pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets): #,pos_weight):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
    
##############################
def main(training_mode):
    
    set_seed(SEED) # setting global random state, as sanity check I also manually set random_state in individual functions    

    ##########
    # messages for user during runtime
    print("Training and crossvalidation code. Reads CSV files from ../data/ folder.")
    print(f"  - random_state seed is set to {SEED}, jumbled flag set to {RUN_JUMBLED}") 
    print(f"  - binder_label is True if abs(Cyp7 Avg, L2FC) > 0 ; used as proxy, any measured property except for PGC works.")    
    print(f"  - coreg_label is True if abs(PGC Avg, L2FC) > {nExpCut}") 
    print(f"  - activity_label is True if ANY abs(PGC Avg, Cyp7 Av, Cyp8 Avg) is > {nExpCut}")    
    ##########
    
    eterms_list=['if_X_fa_atr','if_X_fa_elec','if_X_fa_rep','if_X_fa_sol','if_X_hbond_bb_sc','if_X_hbond_sc']
    
    ## Prepare data  
    expCSV = "../data/malabanan23_si5.csv" # CSV containing experimental measurements
    
    # read energy terms & experimental data. 
    rlDF,_=prepData("../data/out_eterms_IUWtail_7tt8_lid.csv",expCSV)
    # supplement w eterms from 4pld, 7tt8, and full docks of 58 binders
    rlDF=addData_binders(rlDF,"../data/rl/out_eterms_full.csv",expCSV)
    rlDF=addData_binders(rlDF,"../data/rl/out_eterms_4pld.csv",expCSV)
    rlDF=addData_binders(rlDF,"../data/rl/out_eterms_7tt8.csv",expCSV)
    
    # supplement w eterms from db->rl docking of 58 binders
    dbDF = prepDBData('../data/db/out_eterms_7tt8.csv',expCSV,1,'interface_delta_X')
    dbDF = dbDF.reindex(columns=rlDF.columns)
    rlDF = pd.concat([rlDF,dbDF],ignore_index=True)
    
    dbDF = prepDBData('../data/db/out_eterms_full.csv',expCSV,1,'interface_delta_X')    
    dbDF = dbDF.reindex(columns=rlDF.columns)
    rlDF = pd.concat([rlDF,dbDF],ignore_index=True)
    
    dbDF = prepDBData('../data/db/out_eterms_4pld.csv',expCSV,1,'interface_delta_X')
    dbDF = dbDF.reindex(columns=rlDF.columns)
    rlDF = pd.concat([rlDF,dbDF],ignore_index=True)

    #rlDF.to_csv("rlDF_from_train_crossval.csv",index=False) # used for analyses scripts 
    
    ##########
    ## BLiP-L
    ''' 
    Model includes all data from rlDF.
    Train to predict binding. 
    '''
    print("~~~TRAINING BLIP-L")
    
    eterms_np=(rlDF[eterms_list]).to_numpy()
    exp_np=(rlDF['binder_label']).to_numpy()

    if training_mode=="crossval": # peform cross-validation 
        annCrossValid(eterms_np,exp_np.reshape(len(exp_np),1),"binder")
        plt.savefig('result_blipl_crossval.pdf',dpi=300)
    elif training_mode=="save": # train and save a new model     
        annTrainAll(eterms_np,exp_np.reshape(len(exp_np),1),"binder","test_blip-l.pth")

    ##########
    ## ALiP-L
    ''' 
    Model excludes data points from the S2k set with binder prob < rCutBinder (logits). 
    Predict binding using saved model of compounds in rlDF.
    Saves to rlDFBinder with additional columns: binder_pred (binder prediction using BLiP-L) and activity_label (label 0/1). 
    A compound with measured activity effect in any of the 4 experiments in Malabanan 2023 ACS Chem Biol is labeled as an "active" compound. 
    '''
    
    blippth='blip-l.pth'
    print(f"~~~TRAINING ALIP-L, loading {blippth}")
    
    rlDFBinder,rCutBinder=getBinderPred(bliplpth,rlDF,eterms_list) # load previously saved model
    features=eterms_list
    features.append("binder_pred")
    dfBinders = rlDFBinder[rlDFBinder['binder_pred'] > rCutBinder]

    print("   Activators before truncation ", (rlDFBinder['activity_label'] ==1).sum() )
    print("   Activators after truncation ", (dfBinders['activity_label'] ==1).sum() )    
    print("   Dataset ",len(dfBinders))
    
    eterms_np=(dfBinders[features]).to_numpy()
    exp_np=(dfBinders['activity_label']).to_numpy()

    if training_mode=="crossval": # perform cross-validation
        annCrossValid(eterms_np,exp_np.reshape(len(exp_np),1),"activity")
        plt.savefig('result_alipl_crossval.pdf',dpi=300)
    elif training_mode=="save": # train and save a new model 
        annTrainAll(eterms_np,exp_np.reshape(len(exp_np),1),"activity",'test_alip-l.pth') 

    ##########
    
    #plt.show() # show plots
    
    return

##########
def set_seed(seed=None):
    # if set, specifies the random_state seed for any function with random_state
    
    if seed is None:
        return # no random_state seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

##########
def prepData(rldata,expdata,expDF=0,scale=0):
    # if scale==1 : scale IDX to lig eff (=idx/sqrt(# heavy atoms) 

    # RL Energy Terms 
    rlDF=pd.read_csv(rldata)
  
    mwDF = pd.read_csv("../data/compound_properties.csv")
    mwDF = mwDF.set_index('ID')
    mwDF = mwDF.reindex(index=rlDF['ID'])

    ##########
    if (scale): # scale energy terms & IDX by MW
        divcols=['interface_delta_X']

        for index,row in rlDF.iterrows():
            lig=row['ID']
            sq2heavy=math.sqrt(mwDF['nHeavy'][lig])
            rlDF.at[index,divcols] = row[divcols]/sq2heavy 

    ##########
    # get Experiment Data
    if (expdata!="0"): # read csv file, create dataframe 
        expDF=pd.read_csv(expdata,na_values=['#DIV/0!','not determined'])
        expDF.fillna(0,inplace=True)
    else : # expdata read in as a dataframe 
        if not isinstance(expDF, pd.DataFrame):
            sys.exit("Error: the variable is not a dataframe.") 

    rlDF = pd.merge(rlDF, expDF, left_on='ID', right_on='VU Number', how='left').fillna(0)
    rlDF.rename(columns={'VU Number': 'dum'}, inplace=True)
    rlDF.rename(columns={'ID_x': 'VU Number'}, inplace=True)
    rlDF.rename(columns={'ID_y': 'ID'}, inplace=True)
    rlDF.drop('dum',axis=1,inplace=True) 

    rlDF['PGC Avg, L2FC']=rlDF['PGC Avg, L2FC'].astype(float)
    rlDF['PGC std, L2FC']=rlDF['PGC std, L2FC'].astype(float)
    rlDF['Cyp7 Avg, L2FC']=rlDF['Cyp7 Avg, L2FC'].astype(float)
    rlDF['Cyp7 std, L2FC']=rlDF['Cyp7 std, L2FC'].astype(float)
    rlDF['Cyp8 Avg, L2FC']=rlDF['Cyp8 Avg, L2FC'].astype(float)
    rlDF['Cyp8 std, L2FC']=rlDF['Cyp8 std, L2FC'].astype(float)

    rlDF=setLabel(rlDF)

    return rlDF ,expDF
    
##############################
def setLabel(df):

    # Binders
    df['binder_label']= (abs(df['Cyp7 Avg, L2FC']) > 0).astype(int) # any binders
    ## cyp7 is used as a proxy b/c all binders in training/val set have some cyp7 value reported     
    
    # isolated LBD modifiers
    df['coreg_label']= (abs(df['PGC Avg, L2FC']) > nExpCut).astype(int)

    return df

##############################
def addData_binders(df0,rldata,expdata):
    # Read eterms from docks with just original binders 
    # df0 is original df, to which this function will append 
    
    rlDF=pd.read_csv(rldata)

    if (expdata!="0"): # read csv file, create dataframe 
        expDF=pd.read_csv(expdata,na_values=['#DIV/0!','not determined'])
        expDF.fillna(0,inplace=True)
    else : # expdata read in as a dataframe 
        if not isinstance(expDF, pd.DataFrame):
            sys.exit("Error: the variable is not a dataframe.") 

    # merge rl terms & experimental dat to one dataframe
    rlDF = pd.merge(rlDF, expDF, left_on='ID', right_on='ID', how='left').fillna(0)

    rlDF['PGC Avg, L2FC']=rlDF['PGC Avg, L2FC'].astype(float)
    rlDF['PGC std, L2FC']=rlDF['PGC std, L2FC'].astype(float)
    rlDF['Cyp7 Avg, L2FC']=rlDF['Cyp7 Avg, L2FC'].astype(float)
    rlDF['Cyp7 std, L2FC']=rlDF['Cyp7 std, L2FC'].astype(float)
    rlDF['Cyp8 Avg, L2FC']=rlDF['Cyp8 Avg, L2FC'].astype(float)
    rlDF['Cyp8 std, L2FC']=rlDF['Cyp8 std, L2FC'].astype(float)

    # binary label for training 
    rlDF=setLabel(rlDF)          
    rlDF = rlDF.reindex(columns=df0.columns)
    mergedDF = pd.concat([df0,rlDF],ignore_index=True)

    return mergedDF

##############################
def prepDBData(rldata,expdata,saveOne,saveMetric):
    # Read eterms from DB output, which has multiple entries per ligand 

    rlDF=pd.read_csv(rldata)

    if (saveOne): # keep best pose according to saveMetric
        sortDF = rlDF.sort_values(by=['ID', saveMetric])
        rlDF = sortDF.groupby('ID').first().reset_index()

    if (expdata!="0"): # read csv file, create dataframe 
        expDF=pd.read_csv(expdata,na_values=['#DIV/0!','not determined'])
        expDF.fillna(0,inplace=True)
    else : # expdata read in as a dataframe 
        if not isinstance(expDF, pd.DataFrame):
            sys.exit("Error: the variable is not a dataframe.")

    # merge rl terms & experimental dat to one dataframe
    rlDF = pd.merge(rlDF, expDF, left_on='ID', right_on='ID', how='left').fillna(0)

    rlDF['PGC Avg, L2FC']=rlDF['PGC Avg, L2FC'].astype(float)
    rlDF['PGC std, L2FC']=rlDF['PGC std, L2FC'].astype(float)
    rlDF['Cyp7 Avg, L2FC']=rlDF['Cyp7 Avg, L2FC'].astype(float)
    rlDF['Cyp7 std, L2FC']=rlDF['Cyp7 std, L2FC'].astype(float)
    rlDF['Cyp8 Avg, L2FC']=rlDF['Cyp8 Avg, L2FC'].astype(float)
    rlDF['Cyp8 std, L2FC']=rlDF['Cyp8 std, L2FC'].astype(float)

    # binary label for training 
    rlDF=setLabel(rlDF)      
    
    return rlDF 

##########
def annCrossValid(X0,y0,MODE):

    # convert pd df to pytorch tensor
    x0arr=np.array(X0,dtype=np.float32)
    y0arr=np.array(y0,dtype=np.float32)    
    ds=BioData(torch.tensor(x0arr,dtype=torch.float),torch.tensor(y0arr,dtype=torch.float))

    # to preserve true/false ratio
    labels = [outputs for _, outputs in ds]
    
    # split data 
    kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    trues = torch.empty(0,1)
    preds = torch.empty(0,1)

    # for plots
    figloss,axloss=plt.subplots(1,figsize=(5,4))
    figsc,axsc=plt.subplots(1,figsize=(4,4)) 
    fig,ax=plt.subplots(2,figsize=(5,7))
    
    # TRAIN & VALIDATE
    for ifold, (trainIdx, validIdx) in enumerate(kf.split(range(len(ds)),labels)): # stratified
        trainSet = Subset(ds, trainIdx)
        validSet = Subset(ds, validIdx)
        validLoader = DataLoader(validSet, batch_size=64,shuffle=False)
        
        if RUN_JUMBLED:
            trainLoader = DataLoader(jumble_subset(trainSet, seed=ifold), batch_size=64, shuffle=True)
        else:
            trainLoader = DataLoader(trainSet, batch_size=64,shuffle=True)            

        if (MODE=="binder"): # binder prediction WAS RUN W/O MODE
            foldTrainLoss, foldValidLoss, foldTrues, foldPreds , _ = trainModel(trainLoader, valid=validLoader, net=Network(), lr=blipl_params[0], maxEpochs=blipl_params[1])
            
        elif (MODE=="activity"): # regulator/activity prediction
            foldTrainLoss, foldValidLoss, foldTrues, foldPreds , _ = trainModel(trainLoader, valid=validLoader, net=NetworkActivity(), lr=alipl_params[0], maxEpochs=alipl_params[1])
        else :
            sys.exit("ERROR: unknown model MODE (needs to be either `binder' or `activity' ")

        # performance
        print("fold:",ifold)
        axloss.plot([i for i in range(len(foldTrainLoss))],foldTrainLoss,label='Train '+str(ifold))
        axloss.plot([i for i in range(len(foldValidLoss))],foldValidLoss,label='Validation '+str(ifold))
        foldPredsBin = np.where(foldPreds > 0, 1, 0)
        auprc = plotPRC(foldTrues,foldPreds,ax[0],"",'-')
        auroc = plotROC(foldTrues,foldPreds,ax[1],"",'-')
        print(f" AUPRC: {auprc:.2f}")
        print(f" AUROC: {auroc:.2f}")                
        print(f" Precision: {metrics.average_precision_score(foldTrues,foldPreds):.2f}")
        print(f" Recall: {metrics.recall_score(foldTrues,foldPredsBin):.2f}")
        print(f" Accuracy: {metrics.accuracy_score(foldTrues,foldPredsBin):.2f}")
        print(f" MCC: {metrics.matthews_corrcoef(foldTrues,foldPredsBin):.2f}")
        ###
        
    axsc.scatter(foldTrues, torch.sigmoid(foldPreds)) # just for last fold 
    axloss.legend(loc='upper right',frameon=True,facecolor='white',fontsize=11)

    return 
        
##########        
def trainModel(dl,valid=None,net=None, lr=1e-3, maxEpochs=100):
    # ANN training using dl. If valid is set, also determines validation set performance. 
    
    if net is None:
        sys.exit("ERROR: no model network defined")
        
    ######
    N_pos,N_neg=0,0
    for _,label in dl:
        N_pos += (label == 1).sum().item()
        N_neg += (label == 0).sum().item()
    pos_weight = torch.tensor([N_neg/ N_pos], dtype=torch.float32)

    criterion = FocalLoss(alpha=1.0, gamma=1.0, reduction='mean', pos_weight=pos_weight)
    #####

    optimizer = optim.Adam(net.parameters(),lr=lr)
    early_stopper = EarlyStopper(patience=20, min_delta=.0001)

    # use training data to determine mean & std for normalizing 
    batch_samples,tot_samples,tot_sum = 0,0,0
    mean,std = 0.,0.
    for data,_ in dl:
        batch_samples = data.size(0)
        tot_samples += batch_samples
        tot_sum += data.sum(dim=0)
        std += data.std(dim=0) * batch_samples
    mean = tot_sum / tot_samples
    std /= tot_samples

    ## variables to save 
    net.pos_weight.data = pos_weight    
    net.mean.data = mean
    net.std.data = std
    
    # save loss & preds 
    epoch = 0
    trainLoss,validLoss = [],[]
    endPreds, endTrues = None, None
    
    while epoch < maxEpochs:
        running_loss, running_num = 0. , 0
        net.train()
        for data,labels in dl:
            for param in net.parameters():
                param.grad = None
            data_norm = (data - mean) / std
            outputs = net(data_norm)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data_norm.size(0) # scale back bc focal loss reduction is "mean"
            running_num += data_norm.size(0) 
        trainLoss.append(running_loss/running_num)
        
        if valid != None:
            valid_loss, valid_num = 0.,0
            preds, trues = torch.empty(0,1) , torch.empty(0,1)
            net.eval()
            with torch.inference_mode():
                for data,labels in valid:
                    data_norm = (data - mean) / std
                    outputs = net(data_norm)
                    preds = torch.vstack((preds,outputs))
                    trues = torch.vstack((trues,labels))
                    loss = criterion(outputs,labels)
                    valid_loss += loss.item() * data_norm.size(0)
                    valid_num += data_norm.size(0)
                endTrues = trues
                endPreds = preds
            validLoss.append(valid_loss/valid_num)
            if early_stopper.early_stop(valid_loss/valid_num):
                print("early stop at epoch=",epoch) 
                break                                
        epoch +=1
    net.eval()
    
    return trainLoss,validLoss,endTrues, endPreds, net 

###########
def plotPRC(gt,pred,axprc,name,lstyle):

    ax=axprc

    # use prediction probabilities of true class
    precision,recall,thresholds = metrics.precision_recall_curve(gt,pred)
    auprc=metrics.auc(recall,precision)
    ax.plot(recall, precision, label='%s AUPRC=%.3f' % (name,auprc),linestyle=lstyle) #,marker='o')

    # fraction of true positives in dataset is baseline for auprc 
    ftp=sum(gt)/len(gt)
    ax.plot([0, 1], [ftp, ftp], color='k', alpha=0.5 ,lw=0.5, linestyle='dashed')
    
    #plt.title('AUPRC using probabilities of ligand being in "regulator" class')
    ax.legend(loc='upper right',frameon=True,facecolor='white',fontsize=11)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
        
    #plt.text(0.8,0.9,'AUPRC= %.3f' %(auprc))
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ticks=[0,0.2,0.4,0.6,0.8,1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    return auprc
    
##########
def plotROC(gt,pred,axroc,name,lstyle): 
    ax=axroc
    
    fpr, tpr, t = metrics.roc_curve(gt,pred)
    auroc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label='%s AUROC= %.3f' % (name, auroc),linestyle=lstyle) #,marker='o')
    ax.plot([0, 1], [0, 1], color='k', alpha=0.5 ,lw=0.5, linestyle='dashed')
    
    ax.legend(loc='lower right',frameon=True,facecolor='white',fontsize=11)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    #plt.title('ROC from prediction probabilities of ligand being in "activity" class.')

    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ticks=[0,0.2,0.4,0.6,0.8,1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    return auroc

##########
def annTrainAll(X0,y0,MODE,ofname):
        
    x0arr=np.array(X0,dtype=np.float32)
    y0arr=np.array(y0,dtype=np.float32)    
    ds=BioData(torch.tensor(x0arr,dtype=torch.float),torch.tensor(y0arr,dtype=torch.float))

    if (MODE=="binder"): # binder prediction WAS RUN W/O MODE
        trainLoader = DataLoader(ds, batch_size=64,shuffle=True)        
        _,_,_,_, net = trainModel(trainLoader, valid=None, net=Network(), lr=blipl_params[0], maxEpochs=blipl_params[1])
    elif (MODE=="activity"): # regulator/activity prediction
        trainLoader = DataLoader(ds, batch_size=64,shuffle=True)                
        _,_,_,_, net = trainModel(trainLoader, valid=None, net=NetworkActivity(), lr=alipl_params[0], maxEpochs= alipl_params[1])
    else :
        sys.exit("ERROR: unknown model MODE (needs to be either `binder' or `activity' ")

    torch.save(net.state_dict(),ofname) # save model weights and mean/std for normalization  
    
    return 

##########
def getBinderPred(savedNetBinder,df,wts): 
    # gets binder probability prediction from saved model. Saves to df as 'activity_label' and returns 
    
    for index,row in df.iterrows():
        eterms = (row[wts]).to_numpy()
        preds,_ = predict.runTestSet(Network,savedNetBinder,[eterms],1)
        df.at[index, 'binder_pred'] = preds.item()
        
    # for analyzing results
    print(" binder_pred, top 10% cutoff:",df['binder_pred'].quantile(0.90))
    print(" binder_pred, top 30% cutoff:",df['binder_pred'].quantile(0.70))        
    print(" binder_pred, top 50% cutoff:",df['binder_pred'].quantile(0.50))
    binderCut=df['binder_pred'].quantile(0.70)

    ####################
    # add column binary activity label 
    df['activity_label']= (abs(df['PGC Avg, L2FC']) > nExpCut).astype(int)
    df['activity_label'] = (
        (abs(df['PGC Avg, L2FC']) > nExpCut) |
        (abs(df['Cyp7 Avg, L2FC']) > nExpCut) |
        (abs(df['Cyp8 Avg, L2FC']) > nExpCut)
    ).astype(int)    

    return df,binderCut

##########
def getActivityPred(savedNet,df,wts):
    # similar to getBinderPred(savedNetBinder,df,wts), but for activity. Doesn't assign a new label. 
    
    for index,row in df.iterrows():
        eterms = (row[wts]).to_numpy()
        preds,_ = predict.runTestSet(NetworkActivity,savedNet,[eterms],1)
        df.at[index, 'activity_pred'] = preds.item()

    # for analyzing results
    print("activity_pred, top 10% cutoff:",df['activity_pred'].quantile(0.90))
    print("activity_pred, top 30% cutoff:",df['activity_pred'].quantile(0.70))        
    print("activity_pred, top 50% cutoff:",df['activity_pred'].quantile(0.50))

    dfSort=df.sort_values(by='activity_pred').reset_index(drop=True)
    plt.figure()
    plt.scatter(dfSort['activity_label'],dfSort['activity_pred'])
    plt.ylabel('activity_pred')
    plt.xlabel('activity_label')    
    plt.figure()
    plt.scatter(range(len(dfSort)),dfSort['activity_pred'])
    plt.ylabel('activity_pred')
    plt.xlabel('compounds') 
    #plt.show()

    return

##############################
def jumble_subset(subset, features_to_jumble=None, seed=None):
    rng = np.random.default_rng(seed)

    X = subset.dataset.inputs[subset.indices].clone()
    y = subset.dataset.outputs[subset.indices]

    cols = features_to_jumble if features_to_jumble is not None else range(X.shape[1])
    for col in cols:
        X[:, col] = X[rng.permutation(len(X)), col]

    return BioData(X, y)

########################################
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    # set training mode (either perform cros-validation or train and save a new model)
    parser.add_argument("--training_mode", type=str, default="crossval", choices=["crossval", "save"]) 
    parser.add_argument("--jumbled", action="store_true", default=False) # runs jumbled benchmark     

    args = parser.parse_args()    
    SEED = args.seed
    RUN_JUMBLED = args.jumbled and args.training_mode != "save"  # force False if train_save just in case

    main(args.training_mode)
