import dgl
from net.mlp_readout_layer import MLPReadout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import GIN


class GINNet(nn.Module):
    def __init__(self,net_params, *args, **kwargs) -> None:
        super().__init__()

        self.gnn = GIN(
                emb_dim=net_params['hidden_dim'], 
                num_layers=net_params['L'], 
                JK="last", 
                num_node_emb_list=[net_params['num_atom_type']],
                num_edge_emb_list=[net_params['num_bond_type']]
        )
        self.n_classes = net_params.get('n_classes',None)
        self.loss_type = net_params.get('loss_type','l1')
        
        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'bce': # this is for chembl natural targets
            self.loss_fn = nn.BCEWithLogitsLoss(reduction = "none")
        else:
            raise ValueError('loss type not supported')
        
        
        if self.n_classes is not None:
            self.MLP_layer = MLPReadout(net_params['hidden_dim'], self.n_classes)
        else:
            self.MLP_layer = None
    
    def forward(self, g, x, e):
        h = self.gnn(g, [x], [e])
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        out = {'hg': hg}
        if self.n_classes is not None:
            out['scores'] = self.MLP_layer(hg)

        return out
     
    def loss(self, scores, targets):
        if self.loss_type == 'l1':
            loss = self.loss_fn(scores, targets)
        
        elif self.loss_type == 'bce':
            targets = targets.view(scores.shape)
            #Whether y is non-null or not.
            is_valid = targets**2 > 0
            #Loss matrix
            loss_mat = self.loss_fn(scores.double(), (targets+1)/2) # targets are -1,1
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
        
        else:
            raise ValueError('loss type not supported')

        return loss