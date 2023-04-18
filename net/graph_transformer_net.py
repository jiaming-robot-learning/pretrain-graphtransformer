
# adapted from graph transformer https://github.com/graphdeeplearning/graphtransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from .graph_transformer_edge_layer import GraphTransformerLayer
from .mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
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
        
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))

        if self.n_classes is not None:
            self.MLP_layer = MLPReadout(out_dim, self.n_classes)
        else:
            self.MLP_layer = None
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
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

