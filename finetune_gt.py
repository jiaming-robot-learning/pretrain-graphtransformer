"""
    IMPORTING LIBS
"""
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
from utils.metrics import MAE
from net.mlp_readout_layer import MLPReadout
from sklearn.metrics import roc_auc_score

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, None)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    y_true = []
    y_scores = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)

            y_true.append(batch_targets)
            y_scores.append(batch_scores)

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    roc_auc = sum(roc_list)/len(roc_list) #y_true.shape[1]
        
    return epoch_test_loss, epoch_test_mae, roc_auc

def finetune(pretrain_model, dataset, params, net_params, out_dir, exp_name):
    """
    Finetune and evaluate the pre-trained model on the given dataset.
    """

    t0 = time.time()
    per_epoch_time = []
        
    dataset_name = dataset.name
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
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
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = pretrain_model
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
        

    train_loader = DataLoader(trainset, batch_size=params['batch_size'],shuffle=True,num_workers=5, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'],  shuffle=False, num_workers=5,collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'],  shuffle=False, num_workers=5, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                epoch_val_loss, epoch_val_mae, roc_auc_val = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_mae, roc_auc_test = evaluate_network(model, device, test_loader, epoch)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('val/_roc_auc', roc_auc_val, epoch)
                writer.add_scalar('test/_roc_auc', roc_auc_test, epoch)
                # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)


                per_epoch_time.append(time.time()-start)

                # # Saving checkpoint
                # ckpt_dir = f'{out_dir}/checkpoints/{exp_name}'
                # if not os.path.exists(ckpt_dir):
                #     os.makedirs(ckpt_dir)
                # torch.save(model.state_dict(), '{}.pt'.format(ckpt_dir + "/epoch_" + str(epoch)))

                # files = glob.glob(ckpt_dir + '/*.pt')
                # for file in files:
                #     epoch_nb = file.split('_')[-1]
                #     epoch_nb = int(epoch_nb.split('.')[0])
                #     if epoch_nb < epoch-1:
                #         os.remove(file)

                scheduler.step(epoch_val_loss)

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
    
    _, _, roc_auc_test = evaluate_network(model, device, test_loader, epoch)
    print("Test ROC-AUC: {:.4f}".format(roc_auc_test))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(f'{out_dir}/results/{exp_name}.txt', 'w') as f:
        result = f'FINAL RESULTS\nTest ROC-AUC: {roc_auc_test:.4f}\nConvergence Time (Epochs): {epoch:.4f}\n' + \
                f'Total Time Taken: {(time.time()-t0)/3600:.4f} hrs\nAverage Time Per Epoch: {np.mean(per_epoch_time):.4f} s\n\n\n'
        f.write(result)

    return roc_auc_test

def load_weights(ckpt_path, net_params,model):
    """load pretrained model, and change the last layer to match the number of classes in the dataset

    Args:
        ckpt_path (_type_): _description_
        n_class (_type_): _description_
    """
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        # igore the MLP layer
        filtered_dict = {k: v for k, v in checkpoint.items() if k.find('MLP_layer') == -1}
        model.load_state_dict(filtered_dict,strict=False)
        print("=> loaded checkpoint '{}'".format(ckpt_path))
    else:
        print("=> no checkpoint loaded. using random initialization")

    n_class = net_params.get('n_classes',None)
    out_dim = net_params['out_dim']

    model.MLP_layer = MLPReadout(out_dim, n_class)
    
    return model
    
def main():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details",
                        default='config/finetune.json')
    parser.add_argument('--gpu_id', help="Please give a value for gpu id",default=3)
    parser.add_argument('--ckpt', help="Please give a name for pretrained checkpoint",default='pretrain_supervised_zinc_small')
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs",default=100)
    parser.add_argument('--batch_size', help="Please give a value for batch_size",default=32)
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
    # parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    # parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    # parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = f'cuda:{config["gpu"]["id"]}' if config['gpu']['use'] else 'cpu'


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

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')
    if not os.path.exists(out_dir + 'checkpoints'):
        os.makedirs(out_dir + 'checkpoints')

    dataset_names = config['dataset_names']
    ckpt_dir = f'{out_dir}/checkpoints/{args.ckpt}'

    if args.ckpt != '' and os.path.exists(ckpt_dir):
        print(f'Loading checkpoint from {ckpt_dir}')
        os.listdir(ckpt_dir)
        list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(ckpt_dir, x)),
                        os.listdir(ckpt_dir) ) )
        ckpt_path = ckpt_dir + '/' + list_of_files[-1]
    else:
        print(f'No checkpoint found at {ckpt_dir}!')
        ckpt_path = None
        
    results = {}
    for dataset_name in dataset_names:
        print(f'Loading dataset {dataset_name}')
        dataset = load_dataset(dataset_name,config)
        net_params['n_classes'] = dataset.train[0][1].shape[0] # set according to dataset label
        pretrain_model = GraphTransformerNet(net_params)
        pretrain_model = load_weights(ckpt_path,net_params,pretrain_model)
        exp_name = f'finetune_{args.ckpt}_{dataset_name}'

        final_roc = finetune(pretrain_model, dataset, params, net_params, out_dir, exp_name)
        results[dataset_name] = final_roc
        
    
    print(results)
    
    with open(f'{out_dir}/results/finetune_summary_{args.ckpt}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()    
















