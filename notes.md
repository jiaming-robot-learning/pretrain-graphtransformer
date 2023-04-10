
## experiments

### does size matter?
unsupervised zinc, 250k vs 2mil

### supervised or unsupervised? more labels help?
zinc 250k vs chembl_filtered (more labels)

### what training objective works best?
superised
dgi
masking
unsupervised followed by supervised
unsupervised + supervised (together)

### how to pad


### is there a correlation between the features distribution of downstream task and the of the pretriain dataset
use Bhattacharyya distance or Hellinger distance to measure


### supervised chembl with calculated targets
```bash
# logp only
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_logp.json --exp_name pretrain_supervised_chembl_cal_logp --gpu_id 1

# all calculated targets TODO
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_all.json --exp_name pretrain_supervised_chembl_cal_all --gpu_id 1

# selected targets
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_selected.json --exp_name pretrain_supervised_chembl_cal_selected --gpu_id 2

# all natural and calculated targets TODO: bce loss + L1 loss??? ######
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_supervised_chembl_all --dataset chembl_filtered --gpu_id 3

# all natural targets TODO
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_supervised_chembl_natural_labels --dataset chembl_filtered --gpu_id 3
```

## finetune chembl
```bash


```


TODO

pretrain zinc-full
pretrain chembl-selected