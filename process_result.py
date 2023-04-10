
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

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
for d in os.listdir(path):
    if os.path.isdir(path+d) and d.startswith('finetune'):
        exp_name = '_'.join(d.split('_')[1:-1])
        if exp_name not in results:
            results[exp_name] = {}
        db_name = d.split('_')[-1]
        result = {}
        df = parse_tensorboard(path+d, tags)
        v = 0
        for t in tags:
            v += df[t].value[-num_epochs_to_count:].mean()
        results[exp_name][db_name] = v/len(tags)
        
for k in results.keys():
    avg = 0
    for db in results[k].keys():
        avg += results[k][db]
    results[k]['avg'] = avg/len(results[k].keys())

results_df = pd.DataFrame(results)

results_df.to_csv('out/processed_results.csv')
# print(results)
print(results_df)