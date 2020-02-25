import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class SimLoss(nn.Module):
    def __init__(self, tau=1, eps=0.000001):
        super(SimLoss, self).__init__()
        self.tau = tau 
        self.epsilon = eps
    
    def forward(self, batch):
        batch = self.reorder(batch)
        b = torch.mm(batch, batch.transpose(1,0))
        norm = torch.norm(batch, p=2, dim=1).unsqueeze(1)
        norm = torch.mm(norm, norm.transpose(1,0))
        den = (torch.ones(batch.shape[0]) - torch.eye(batch.shape[0])).cuda() * b / (norm + self.epsilon) # add eps for numerical stability
        den = torch.sum(torch.exp(den/self.tau), dim=1)
        
        num = torch.zeros(batch.shape[0]).float().cuda()
        for k in range(batch.shape[0]):
            i, j = (k//2)*2, (k//2)*2+1
            num[k] = b[i][j] / norm[i][j]
        loss = torch.sum(-torch.log(torch.exp(num/self.tau) / den))/batch.shape[0]
        
        return loss
   
    def reorder(self, batch):
        b = torch.zeros_like(batch)
        for i in range(batch.shape[0]//2):
            b[2*i] = batch[i]
            b[2*i+1] = batch[i+(batch.shape[0]//2)]
        return b

    

