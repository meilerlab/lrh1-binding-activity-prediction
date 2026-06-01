import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Generate the dataset
class BioData(Dataset):
    def __init__(self, X0,y0):
        self.inputs=X0
        self.outputs=y0
    def __len__(self):
        return self.outputs.size(0)
    def __getitem__(self, idx):
        return (self.inputs[idx,:], self.outputs[idx,:])

class Network(nn.Module): 
    def __init__(self):
        super().__init__()
        self.linin = nn.Linear(6,24)
        self.layer1 = nn.Linear(24,6)
        self.layer2 = nn.Linear(6,1)  
        self.relu = nn.LeakyReLU() 
        self.fa = nn.Sigmoid()
        # save param for z-normalization & loss 
        self.mean = nn.Parameter(torch.zeros(6), requires_grad=False)
        self.std = nn.Parameter(torch.ones(6), requires_grad=False)
        self.pos_weight = nn.Parameter(torch.tensor(1), requires_grad=False)
        
    def forward(self,x):
        x = self.linin(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        #x = self.fa(x) # no activation (using raw logits)
        return x 

class NetworkActivity(nn.Module):
    def __init__(self):
        super().__init__()
        self.linin = nn.Linear(7,24)
        self.layer1 = nn.Linear(24,7)
        self.layer2 = nn.Linear(7,1)
        self.relu = nn.LeakyReLU()
        self.fa = nn.Sigmoid()
        # save param for z-normalization & loss 
        self.mean = nn.Parameter(torch.zeros(7), requires_grad=False)
        self.std = nn.Parameter(torch.ones(7), requires_grad=False)
        self.pos_weight = nn.Parameter(torch.tensor(1), requires_grad=False)
    def forward(self,x):
        x = self.linin(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        #x = self.fa(x) # no activation (using raw logits)        
        return x
