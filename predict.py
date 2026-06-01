## function to get predictions from models 

import numpy as np 
import torch
from utils_define_data_model import BioData

def runTestSet(modelClass,ifnModel,X0,y0=None):
    # Use saved ANN to predict ouptuts

    x0arr=np.array(X0,dtype=np.float32)
    y0arr=np.array(y0,dtype=np.float32)    
    dtest=BioData(torch.tensor(x0arr,dtype=torch.float),torch.tensor(y0arr,dtype=torch.float) if y0 is not None else None)

    net = modelClass()
    net.load_state_dict(torch.load(ifnModel,weights_only=True),strict=False)
    net.eval()

    with torch.inference_mode():
        mean,std = net.mean.detach(), net.std.detach()
        data = dtest.inputs
        data_norm =  (data - mean) / std        
        outputs = net(data_norm) # outputs logits; see train_crossval_save.py
    
    preds = outputs.ravel()
    predsBin = np.where(preds > 0, 1, 0)     

    return preds,predsBin
