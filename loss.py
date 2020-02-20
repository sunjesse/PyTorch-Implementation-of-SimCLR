import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def pairwise_sim(self, z_i, z_j):
        c = f.cosine_similarity(z_i, z_j, dim=0).float()
        return c
    
    def soft_nll(self, i, j, batch):
        s_ij = torch.exp(self.pairwise_sim(batch[i-1], batch[j-1]))
        s_ik = torch.zeros(batch.shape[0])
        for k in range(batch.shape[0]):
            if k != i-1:
                s_ik = torch.exp(self.pairwise_sim(batch[i-1], batch[k]))

        return -torch.log(s_ij/torch.sum(s_ik))

    def forward(self, batch):
        loss = torch.zeros(batch.shape[0]).float()
        for k in range(1, batch.shape[0] // 2 + 1):
            loss[k] = self.soft_nll(2*k-1, 2*k, batch) + self.soft_nll(2*k, 2*k-1, batch)
        loss = torch.sum(loss) / loss.shape[0]
        return loss
