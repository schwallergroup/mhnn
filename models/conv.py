import torch
import torch.nn as nn
from torch_scatter import scatter
from models.mlp import MLP


class MHNNConv(nn.Module):
    def __init__(self, hid_dim, mlp1_layers=1, mlp2_layers=1, mlp3_layers=1,
        mlp4_layers=1, aggr='mean', dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(hid_dim*2, hid_dim, hid_dim, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = lambda X: X[..., hid_dim:]

        if mlp2_layers > 0:
            self.W2 = MLP(hid_dim*2, hid_dim, hid_dim, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., hid_dim:]

        if mlp3_layers > 0:
            self.W3 = MLP(hid_dim*2, hid_dim, hid_dim, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W3 = lambda X: X[..., hid_dim:]

        if mlp4_layers > 0:
            self.W4 = MLP(hid_dim*2, hid_dim, hid_dim, mlp4_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W4 = lambda X: X[..., hid_dim:]
        self.aggr = aggr
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()
        if isinstance(self.W4, MLP):
            self.W4.reset_parameters()

    def forward(self, X, E, vertex, edges):
        N = X.shape[-2]

        Mve = self.W1(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Me = scatter(Mve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        E = self.W2(torch.cat((E, Me), -1))
        # E = E*0.5 + e_in*0.5  # Residual connection.
        Mev = self.W3(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Mv = scatter(Mev, vertex, dim=-2, reduce=self.aggr, dim_size=N)
        X = self.W4(torch.cat((X, Mv), -1))
        # X = X*0.5 + X0*0.5  # Residual connection.

        return X, E


class MHNNSConv(nn.Module):
    def __init__(self, hid_dim, mlp1_layers=1, mlp2_layers=1, mlp3_layers=1,
                 aggr='mean', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(hid_dim, hid_dim, hid_dim, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(hid_dim*2, hid_dim, hid_dim, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., hid_dim:]

        if mlp3_layers > 0:
            self.W3 = MLP(hid_dim, hid_dim, hid_dim, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :] # [nnz, C]
        Xe = scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C]
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = (1-self.alpha) * Xv + self.alpha * X0
        X = self.W3(X)

        return X
