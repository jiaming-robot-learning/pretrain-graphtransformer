


# # pretrain zinc-full
# python pretrain_gt_supervised.py --config config/pretrain_supervised.json --exp_name pretrain_supervised_zinc_full --dataset zinc_full --gpu_id 3

# # pretrain chembl-selected this corresponds to zinc 3 labels
# python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_selected.json --exp_name pretrain_supervised_chembl_syn_selected --dataset chembl_filtered --gpu_id 3

# # pretrain chembl-selected-plus
# python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_selected_plus.json --exp_name pretrain_supervised_chembl_syn_selected_plus --dataset chembl_filtered --gpu_id 3

# # pretrain chembl-cal_all? need to fix nan and inf issues

# # pretrain dgi_zinc_small
# python pretrain_gt_dgi.py --config config/pretrain_dgi.json --exp_name pretrain_dgi_zinc_small --dataset zinc_small --gpu_id 3

# # pretrain masking_zinc_small
# python pretrain_gt_masking.py --config config/pretrain_masking.json --exp_name pretrain_masking_zinc_small --dataset zinc_small --gpu_id 3

# # pretrain supervised_chembel_natural
# python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_supervised_chembl_natural_labels --dataset chembl_filtered --gpu_id 3


# # pretrain supervised_zinc_small
# python pretrain_gt_supervised.py --config config/pretrain_supervised.json --exp_name pretrain_supervised_zinc_small --dataset zinc_small --gpu_id 3


# ##### unsup -> supervised
# python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_dgi_supervised_zinc_small --dataset chembl_filtered --gpu_id 3 --ckpt pretrain_dgi_zinc_small
# python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_masking_supervised_zinc_small --dataset chembl_filtered --gpu_id 3 --ckpt pretrain_masking_zinc_small


# #### unsup + sup
# python pretrain_gt_dgi.py --config config/pretrain_dgi.json --exp_name pretrain_dgisup05_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 0.5
# python pretrain_gt_dgi.py --config config/pretrain_dgi.json --exp_name pretrain_dgisup1_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 1
# python pretrain_gt_dgi.py --config config/pretrain_dgi.json --exp_name pretrain_dgisup2_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 2


# ##### gin
# python pretrain_gin_supervised.py --config config/pretrain_supervised.json --exp_name pretrain_gin_supervised_zinc_small --dataset zinc_small --gpu_id 3

# python pretrain_gin_supervised.py --config config/pretrain_supervised.json --exp_name pretrain_gin_supervised_zinc_full --dataset zinc_full --gpu_id 3


# python pretrain_gin_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_gin_supervised_chembl_natural_labels --dataset chembl_filtered --gpu_id 3

# # gin unsup
# python pretrain_gin_masking.py --config config/pretrain_masking.json --exp_name pretrain_gin_masking_zinc_full --dataset zinc_full --gpu_id 3

# # then sup using chembl labels
# python pretrain_gin_supervised.py --config config/pretrain_supervised_chembl_natural_labels.json --exp_name pretrain_gin_masking_zincfull_supervised_chembl --dataset chembl_filtered --gpu_id 3 --ckpt pretrain_gin_masking_zinc_full

# # unsup -> sup with calculated labels 
# python pretrain_gin_supervised.py --config config/pretrain_supervised.json --exp_name pretrain_gin_masking_zincfull_supervised_zincfull --dataset zinc_full --gpu_id 3 --ckpt pretrain_gin_masking_zinc_full

# # directly unsup + sup
# python pretrain_gin_masking.py --config config/pretrain_masking.json --exp_name pretrain_gin_maskingsup1_zinc_full --dataset zinc_full --gpu_id 2 --sup_ratio 1


##### gt with nope, rwpe and small size
# python pretrain_gt_supervised.py --config config/pretrain_supervised_nope.json --exp_name pretrain_gtnope_supervised_zinc_small --dataset zinc_small --gpu_id 3

# python pretrain_gt_supervised.py --config config/pretrain_supervised_rwpe.json --exp_name pretrain_gtrwpe_supervised_zinc_small --dataset zinc_small --gpu_id 3

# python pretrain_gt_supervised.py --config config/pretrain_supervised_small.json --exp_name pretrain_gtsmall_supervised_zinc_small --dataset zinc_small --gpu_id 3

# python pretrain_gt_supervised.py --config config/pretrain_supervised_nopesmall.json --exp_name pretrain_gtnopesmall_supervised_zinc_small --dataset zinc_small --gpu_id 3

# python pretrain_gt_supervised.py --config config/pretrain_supervised_nopesmaller.json --exp_name pretrain_gtnopesmall_supervised_zinc_smaller --dataset zinc_small --gpu_id 3




##### gt re-train with smaller size no pe

# pretrain zinc-full
python pretrain_gt_supervised.py --config config/pretrain_supervised_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_supervised_zinc_full --dataset zinc_full --gpu_id 3 --load_pyg_dataset True

# pretrain chembl-selected this corresponds to zinc 3 labels
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_selected_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_supervised_chembl_syn_selected --dataset chembl_filtered --gpu_id 3

# pretrain chembl-selected-plus
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_syn_selected_plus_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_supervised_chembl_syn_selected_plus --dataset chembl_filtered --gpu_id 3

# pretrain dgi_zinc_small
python pretrain_gt_dgi.py --config config/pretrain_dgi_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_dgi_zinc_small --dataset zinc_small --gpu_id 3

# pretrain masking_zinc_small
python pretrain_gt_masking.py --config config/pretrain_masking_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_masking_zinc_small --dataset zinc_small --gpu_id 3

# pretrain supervised_chembel_natural
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_supervised_chembl_natural_labels --dataset chembl_filtered --gpu_id 3


# pretrain supervised_zinc_small
python pretrain_gt_supervised.py --config config/pretrain_supervised_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_supervised_zinc_small --dataset zinc_small --gpu_id 3


##### unsup -> supervised
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_dgi_supervised_zinc_small --dataset chembl_filtered --gpu_id 3 --ckpt pretrain_gtrwpesmall_dgi_zinc_small
python pretrain_gt_supervised.py --config config/pretrain_supervised_chembl_natural_labels_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_masking_supervised_zinc_small --dataset chembl_filtered --gpu_id 3 --ckpt pretrain_gtrwpesmall_masking_zinc_small


#### unsup + sup
python pretrain_gt_dgi.py --config config/pretrain_dgi_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_dgisup05_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 0.5
python pretrain_gt_dgi.py --config config/pretrain_dgi_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_dgisup1_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 1
python pretrain_gt_dgi.py --config config/pretrain_dgi_gtrwpesmall.json --exp_name pretrain_gtrwpesmall_dgisup2_zinc_small --dataset zinc_small --gpu_id 2 --sup_ratio 2