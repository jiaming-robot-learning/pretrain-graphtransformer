
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from net.graph_transformer_net import GraphTransformerNet
from utils.dataset_dgl import load_dataset

from utils.dataset_pyg import MoleculeDatasetG, allowable_features
from train.train_molecules_graph_regression import train_epoch, evaluate_network

def pretrain(dataset, params, net_params, out_dir, exp_name):
    """
    pretrain the model using specified dataset.
    We don't use the validation set here, because we evaluate the model on the downstream task.
    """

    t0 = time.time()
    per_epoch_time = []
        
    dataset_name = dataset.name
    trainset = dataset.train
        
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(f'{out_dir}/configs/{exp_name}.txt', 'w') as f:
        f.write(f'Dataset: {dataset_name},\nparams={params}\n\n' +
                f'nnet_params={net_params}\n\n')
        
    writer = SummaryWriter(log_dir=f'{out_dir}/logs/{exp_name}')

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))

    model = GraphTransformerNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses  = []
    epoch_train_MAEs  = []
        
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], num_workers=8, shuffle=True, collate_fn=dataset.collate)
    
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)
                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                epoch_train_losses.append(epoch_train_loss)
                epoch_train_MAEs.append(epoch_train_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, 
                              train_MAE=epoch_train_mae)
                              
                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = f'{out_dir}/checkpoints/{exp_name}'
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pt'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pt')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_train_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, train_mae = evaluate_network(model, device, train_loader, epoch)
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(f'{out_dir}/results/{exp_name}.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(dataset_name,  params, net_params, model, net_params['total_param'],
                   train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        

def main():    
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details",
                        default='config/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json')
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name",
                        default='zinc_small')
    parser.add_argument('--out_dir', help="Please give a value for out_dir",default='out/')
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")

    # parser.add_argument('--batch_size', help="Please give a value for batch_size")
    # parser.add_argument('--init_lr', help="Please give a value for init_lr")
    # parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    # parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    # parser.add_argument('--min_lr', help="Please give a value for min_lr")
    # parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    # parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    # parser.add_argument('--L', help="Please give a value for L")
    # parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    # parser.add_argument('--out_dim', help="Please give a value for out_dim")
    # parser.add_argument('--residual', help="Please give a value for residual")
    # parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    # parser.add_argument('--readout', help="Please give a value for readout")
    # parser.add_argument('--n_heads', help="Please give a value for n_heads")
    # parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    # parser.add_argument('--dropout', help="Please give a value for dropout")
    # parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    # parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    # parser.add_argument('--self_loop', help="Please give a value for self_loop")
    # parser.add_argument('--max_time', help="Please give a value for max_time")
    # parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim" )
    # parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    # parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = f'cuda:{config["gpu"]["id"]}' if config['gpu']['use'] else 'cpu'
  
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    # if args.batch_size is not None:
    #     params['batch_size'] = int(args.batch_size)
    # if args.init_lr is not None:
    #     params['init_lr'] = float(args.init_lr)
    # if args.lr_reduce_factor is not None:
    #     params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    # if args.lr_schedule_patience is not None:
    #     params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    # if args.min_lr is not None:
    #     params['min_lr'] = float(args.min_lr)
    # if args.weight_decay is not None:
    #     params['weight_decay'] = float(args.weight_decay)
    # if args.print_epoch_interval is not None:
    #     params['print_epoch_interval'] = int(args.print_epoch_interval)
    # if args.max_time is not None:
    #     params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    # net_params['batch_size'] = params['batch_size']
    # if args.L is not None:
    #     net_params['L'] = int(args.L)
    # if args.hidden_dim is not None:
    #     net_params['hidden_dim'] = int(args.hidden_dim)
    # if args.out_dim is not None:
    #     net_params['out_dim'] = int(args.out_dim)   
    # if args.residual is not None:
    #     net_params['residual'] = True if args.residual=='True' else False
    # if args.edge_feat is not None:
    #     net_params['edge_feat'] = True if args.edge_feat=='True' else False
    # if args.readout is not None:
    #     net_params['readout'] = args.readout
    # if args.n_heads is not None:
    #     net_params['n_heads'] = int(args.n_heads)
    # if args.in_feat_dropout is not None:
    #     net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    # if args.dropout is not None:
    #     net_params['dropout'] = float(args.dropout)
    # if args.layer_norm is not None:
    #     net_params['layer_norm'] = True if args.layer_norm=='True' else False
    # if args.batch_norm is not None:
    #     net_params['batch_norm'] = True if args.batch_norm=='True' else False
    # if args.self_loop is not None:
    #     net_params['self_loop'] = True if args.self_loop=='True' else False
    # if args.lap_pos_enc is not None:
    #     net_params['lap_pos_enc'] = True if args.pos_enc=='True' else False
    # if args.pos_enc_dim is not None:
    #     net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    # if args.wl_pos_enc is not None:
    #     net_params['wl_pos_enc'] = True if args.pos_enc=='True' else False
        

    net_params['num_atom_type'] = len(allowable_features['possible_atomic_num_list'])  
    net_params['num_bond_type'] = len(allowable_features['possible_bonds'])  

    dataset = load_dataset(DATASET_NAME,config)
    net_params['n_classes'] = dataset.train[0][1].shape[0] # set according to dataset label

    exp_name = 'pretrain' +  "_" + DATASET_NAME

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    if not os.path.exists(out_dir + 'checkpoints'):
        os.makedirs(out_dir + 'checkpoints')

    pretrain( dataset, params, net_params, out_dir, exp_name)
    
    
main()    
















