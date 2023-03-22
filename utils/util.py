import torch


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
    
    
def print_dataset(dataset):
    ys = [0,0]
    t_n = 0
    node_n = 0
    edge_n = 0
    for i, d in enumerate(dataset):
        ys[int(d.y.item())] += 1
        t_n += 1
        node_n += d.x.shape[0]
        edge_n += d.edge_index.shape[1] // 2

    print(f'Total number of graphs in dataset: {t_n}.')
    print(f'Num of positive labels: {ys[1]} - {ys[1]/t_n:.2%}')
    print(f'Num of negative labels: {ys[0]} - {ys[0]/t_n:.2%}')
    print(f'Avg. num of nodes per graph: {node_n/t_n:.2f}')
    print(f'Avg. num of edges per graph: {edge_n/t_n:.2f}')
    print(f'Num of node features: {d.x.shape[1]}')
    print(f'Num of edge features: {d.edge_attr.shape[1]}')

    d = dataset[0]
    print(d)