import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
        self.weight1 = nn.Parameter(torch.ones(1))
        
    def l2_softmax(self, x, alpha):
        l2 = torch.sqrt((x**2).sum())
        x = alpha * (x / l2)
        return x
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            
        pre = y
        
        #alpha = self.weight1
        #y = self.l2_softmax(y, alpha)
        y = self.FC_layers[self.L](y)
        return y