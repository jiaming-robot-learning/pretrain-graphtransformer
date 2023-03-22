
import argparse
import json
from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

from utils.splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from net.graph_transformer_net import GraphTransformerNet
from torch_geometric.datasets import ZINC

from tensorboardX import SummaryWriter
from utils.util import print_dataset

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class Infomax(nn.Module):
    def __init__(self, gnn, discriminator):
        super(Infomax, self).__init__()
        self.gnn = gnn
        self.discriminator = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_mean_pool


def train(args, model, device, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
        summary_emb = torch.sigmoid(model.pool(node_emb, batch.batch))

        positive_expanded_summary_emb = summary_emb[batch.batch]

        shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)]
        negative_expanded_summary_emb = shifted_summary_emb[batch.batch]

        positive_score = model.discriminator(node_emb, positive_expanded_summary_emb)
        negative_score = model.discriminator(node_emb, negative_expanded_summary_emb)      

        optimizer.zero_grad()
        loss = model.loss(positive_score, torch.ones_like(positive_score)) + model.loss(negative_score, torch.zeros_like(negative_score))
        loss.backward()

        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/step, train_loss_accum/step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=10,
                        help='number of GNN message passing layers (default: 10).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    # parser.add_argument('--dataset', type=str, default = 'hiv', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--config', type=str, default = 'config/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json', help='config file for infomax')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=False,split='val')
    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=False,split='train')
    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=False,split='test')
    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=True,split='val')
    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=True,split='train')
    # zinc_data = ZINC(root='dataset/zinc_geometric',subset=True,split='test')
    # print_dataset(zinc_data)

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    # with open(args.config) as f:
    #     config = json.load(f)
    # gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    net_params = {
        "L": args.num_layer,
        "n_heads": 8,
        "hidden_dim": args.emb_dim, # same as pretrain-gnns
        "out_dim": 64,
        "edge_feat": True,
        "residual": True,
        "readout": "mean",
        "in_feat_dropout": 0.0,
        "dropout": args.dropout_ratio,
        "layer_norm": False,
        "batch_norm": True,
        "self_loop": False,
        "lap_pos_enc": True,
        "pos_enc_dim": 8,
        "wl_pos_enc": False,
        "full_graph": False,
        "n_classes": 1,
        'device': 'cuda:0',
        'num_atom_type': 119,
        'num_bond_type': 4,
    }
    gnn = GraphTransformerNet(net_params)

    discriminator = Discriminator(args.emb_dim)
    model = Infomax(gnn, discriminator)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, loader, optimizer)

        print(train_acc)
        print(train_loss)


    if not args.output_model_file == "":
        torch.save(gnn.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()

