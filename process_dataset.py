from net.graph_transformer_net import GraphTransformerNet
from utils.dataset_dgl import process_all_dataset,print_stats_all_dataset
from typing import Dict
from utils.dictionary import Dictionary

def main():
    process_all_dataset()
    # print_stats_all_dataset()


def load_dict():
    import pickle
    with open('dataset/zinc_geometric/raw/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('dataset/zinc_geometric/raw/atom_dict.pickle', 'rb') as f:
        atom_dict = pickle.load(f)
        print(atom_dict)
    with open('dataset/zinc_geometric/raw/bond.pickle', 'rb') as f:
        bond_dict = pickle.load(f)
    
# run main if this file is called directly
if __name__ == "__main__":
    # load_dict()
    main()