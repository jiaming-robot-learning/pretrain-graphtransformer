
# adapted from
import torch
import dill 
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from utils.splitters import scaffold_split
import pandas as pd
from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
from utils.dataset_pyg import MoleculeDatasetG, allowable_features
import threading
from dgl.data.utils import save_graphs, load_graphs

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adj(scipy_fmt='coo').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class MoleculeDataset(torch.utils.data.Dataset):
    """This is the actual dataset class that will be used to load the data.
    Load pyG dataset and convert to DGLGraph

    Args:
        torch (_type_): _description_
    """
    def __init__(self, name='zinc_dataset_full'):
        """
            Loading ZINC dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        empty_dataset = MoleculeDGL()
        datasets = []
        for split in ['train','val','test']:
            d = MoleculeDGL()
            d.load(name,split)
            datasets.append(d)
        
        self.train, self.val, self.test = datasets

        # fname = 'dataset/' + name + '/processed/dataset_dgl.pickle'
        # with open(fname, 'rb') as f:
        #     dataset = pickle.load(f)
        # # fname = f'dataset/{name}/processed/dataset_dgl.pickle'
        # # with open(fname, "rb") as f:
        # #     f = pickle.load(f)
        #     self.train = dataset.train
        #     self.val = dataset.val if dataset.val is not None else empty_dataset
        #     self.test = dataset.test if dataset.test is not None else empty_dataset
        #     self.num_atom_type = dataset.num_atom_type
        #     self.num_bond_type = dataset.num_bond_type
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        # labels = torch.stack(labels,dim=0)
        labels = torch.concat(labels).unsqueeze(1)
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]
    
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]


class MoleculeDGL(torch.utils.data.Dataset):
    """This class is used to load the pyG dataset and convert it to DGL graphs.

    Args:
        torch (_type_): _description_
    """
    def __init__(self, data=None,n_thread=10):
        
        self.graph_lists = []
        self.graph_labels = []

        if data is not None:
            self.data = data
            self.n_thread = n_thread
            self._prepare()
        
    def save(self, name,split):
        if self.graph_labels[0] is None:
            labels = None
        else:
            labels = {'glabel': torch.stack(self.graph_labels,dim=0)}
        save_graphs(f'dataset/{name}/processed/dataset_dgl_{split}.pkl', self.graph_lists, labels=labels)

    def load(self,name,split):
        fname = f'dataset/{name}/processed/dataset_dgl_{split}.pkl'
        if os.path.exists(fname):
            print(f'Loading dataset from {fname}...')
            self.graph_lists, self.graph_labels = load_graphs(f'dataset/{name}/processed/dataset_dgl_{split}.pkl')
            self.graph_labels = list(self.graph_labels['glabel'])
        else:
            print(f'File {fname} doesnot exist. Init empty dataset instead.')

    def _prepare(self):
        
        index = [[i for i in range(len(self.data)) if i % self.n_thread == j] for j in range(self.n_thread)]
        graph_list = [None for i in range(len(self.data))]
        label_list = [None for i in range(len(self.data))]
        
        def _prepare_thread(data,index,graph_list,label_list):
            for idx in index:
                molecule = data[idx]
                node_features = molecule.x.long()[:,0] # only using the atom type
                edge_features = molecule.edge_attr.long()[:,0] # only using the bond type
                
                # Create the DGL Graph
                g = dgl.DGLGraph()
                g.add_nodes(molecule.x.shape[0])
                g.ndata['feat'] = node_features

                g.add_edges(molecule.edge_index[0,:], molecule.edge_index[1,:])
                # for i in range(molecule.edge_index.shape[1]):
                #     g.add_edges(molecule.edge_index[0,i].item(), molecule.edge_index[1,i].item())

                g.edata['feat'] = edge_features
            
                graph_list[idx] = g
                label_list[idx] = molecule.y
                
                
        threads = [threading.Thread(target=_prepare_thread, args=(self.data,index[i],graph_list,label_list)) for i in range(self.n_thread)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        self.graph_lists = graph_list
        self.graph_labels = label_list
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeDatasetDGL(object):
    """This class is used to save the dataset in a pickle file

    Args:
        object (_type_): _description_
    """
    def __init__(self, name, train_data, val_data, test_data):
        
        self.name = name
        
        self.num_atom_type = len(allowable_features['possible_atomic_num_list'])  # 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = len(allowable_features['possible_bonds'])  # 4 # known meta-info about the zinc dataset; can be calculated as well
        
        self.train = train_data
        self.val = val_data
        self.test = test_data
        
def process_all_dataset(reprocess=False):
    """Create dgl graph datasets

    Args:
        reprocess (bool, optional): _description_. Defaults to False.
    """
    pretrain_dataset = [
            'zinc_standard_agent',
            'chembl_filtered',
            ]

    for dataset_name in pretrain_dataset:
        # load G dataset
        print(dataset_name)
        root = "dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        if reprocess and os.path.exists(root + "/processed/geometric_data_processed.pt"):
            os.remove(root + "/processed/geometric_data_processed.pt")
        dataset = MoleculeDatasetG(root, dataset=dataset_name)
        print(f"G dataset loaded {dataset_name}")
        print(dataset)

        # convert to dgl graph
        dataset_dgl_train = MoleculeDGL(dataset)
        dataset_dgl_train.save(dataset_name,'train')

        # # save dgl graph
        # dataset_dgl_all = MoleculeDatasetDGL(dataset_name, dataset_dgl_train,None,None)
        # with open(root + "/processed/dataset_dgl.pickle", 'wb') as f:
        #     pickle.dump(dataset_dgl_all, f,protocol=pickle.HIGHEST_PROTOCOL)
        
    downstream_dir = [
            'bace',
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'muv',
            'sider',
            'tox21',
            'toxcast',
            ]

    for dataset_name in downstream_dir:
        # first load G dataset
        print(dataset_name)
        root = "dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        if reprocess and os.path.exists(root + "/processed/geometric_data_processed.pt"):
            os.remove(root + "/processed/geometric_data_processed.pt")
        dataset = MoleculeDatasetG(root, dataset=dataset_name)
        print(f"G dataset loaded {dataset_name}")
        print(dataset)
        
        # split G dataset
        print(f'Splitting dataset...')
        smiles_list = pd.read_csv('dataset/' + dataset_name + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print(f'dataset splitted')

        # convert DGL dataset for all
        dataset_dgl_train = MoleculeDGL(train_dataset)
        print(f'train dataset converted to DGL')
        dataset_dgl_train.save(dataset_name,'train')

        dataset_dgl_val = MoleculeDGL(valid_dataset)
        print(f'val dataset converted to DGL')
        dataset_dgl_val.save(dataset_name,'val')

        dataset_dgl_test = MoleculeDGL(test_dataset)
        print(f'test dataset converted to DGL')
        dataset_dgl_test.save(dataset_name,'test')
        
        # create DGL dataset and save
        # dataset_dgl_all = MoleculeDatasetDGL(dataset_name, dataset_dgl_train, dataset_dgl_val, dataset_dgl_test)
        # with open(root + "/processed/dataset_dgl.pickle", 'wb') as f:
        #     s = pickle.dumps(dataset_dgl_all,protocol=pickle.HIGHEST_PROTOCOL)
        #     # pickle.dump(dataset_dgl_all, f,protocol=pickle.HIGHEST_PROTOCOL)
        #     f.write(s)
            
        
        print(f'DGL dataset saved {dataset_name}')
        
def load_dataset(name):
    """
    Load dataset from pickle file
    Can be one of the following:
            ['bace',
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'muv',
            'sider',
            'tox21',
            'toxcast',
            'chembl_filtered',
            'zinc_standard_agent'
            ]
    """
    
    return MoleculeDataset('bbbp')

# def preprocess_zinc():
#     zinc_dataset = MoleculeDatasetDGL(name='ZINC-full')
#     with open('dataset/zinc_geometric/processed/zinc_dataset_full.pickle', 'wb') as f:
#         pickle.dump(zinc_dataset, f)
#     print("Zinc dataset is ready!")
