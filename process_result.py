
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import numpy as np
import json

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


path = f'out/logs/'
tags = ['test/_roc_auc','val/_roc_auc']
num_epochs_to_count = 5
results = {}
residue_all = {}
pretrain_datasets_tmp = {
    'zinc_small':'zinc_small',
    'zinc_full':"zinc_full",
    'zincfull':'zinc_full',
    'chembl':'chembl_filtered',
    }
pretrain_datasets = {
    'zinc_small':'ZINC 250k',
    'zinc_full':"ZINC 1M",
    'zincfull':'ZINC 1M',
    'chembl':'ChemBL',
    }

no_pretrain = {
}
for d in os.listdir(path):
    if os.path.isdir(path+d) and d.startswith('finetune'):
        exp_name = '_'.join(d.split('_')[1:-1])
        if exp_name not in results:
            results[exp_name] = {}
        db_name = d.split('_')[-1]
        result = {}
        df = parse_tensorboard(path+d, tags)
        v = 0
        residual_tmp = 0
        for t in tags:
            v += df[t].value[-num_epochs_to_count:].mean()
            original = df[t].value
            smoothed = smooth(df[t].value, 0.6)
            
            residuals = [abs(x - y) for x, y in zip(original, smoothed)]
            residual = np.mean(residuals)
            residual_tmp += residual
                
        results[exp_name][db_name] = v/len(tags)

        residue_all[exp_name] = residual_tmp/len(tags)
        
for k in results.keys():
    avg = 0
    for db in results[k].keys():
        avg += results[k][db]
    results[k]['avg'] = avg/len(results[k].keys())
    results[k]['residual'] = residue_all[k]

    for d,dv in pretrain_datasets.items():
        if k.find(d) != -1:
            results[k]['pretrain_dataset'] = dv
            results[k]['pretrain_dataset_tmp'] = pretrain_datasets_tmp[d]
            break

results_df = pd.DataFrame(results)
results_df = results_df.transpose()

with open('out/dataset_stats/hellinger_dist.json', 'r') as f:
    hellinger_dist = json.load(f)

hd = {}
for k,v in hellinger_dist.items():
    k1,k2 = k.split('-')
    if k1 not in hd:
        hd[k1] = {}
    hd[k1][k2] = v
    
hd_df = pd.DataFrame(hd)
hd_df.to_csv('out/dataset_stats/hd.csv')

# calculate the correlation between rows of results and rows of hellinger distance for columns in the list
columns = ['clintox','muv','bace','bbbp','hiv','sider','tox21','toxcast']

for i in range(len(results_df)):
    row = results_df.iloc[i]
    if results_df.index[i].find('gin') != -1:
        pretrain_row =results_df.loc['gin_no_pretrain',columns].astype(float)
    else:
        pretrain_row =results_df.loc['no_pretrain',columns].astype(float)
        
    pretrain_dataset = row['pretrain_dataset_tmp']
    if pretrain_dataset not in hd_df.index:
        continue
    improvements = row[columns].astype(float) - pretrain_row
    corr = improvements.corr(hd_df.loc[pretrain_dataset,columns])
    results_df.loc[results_df.index[i],'corr'] = corr
    
print(results_df)
results_df.to_csv('out/processed_results.csv')