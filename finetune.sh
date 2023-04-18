

# check already finetuned models
finetuned=()
for result in out/results/*; do
 if [[ "${result##*/}" =~ finetune_summary.* ]]; then
    name="${result##*summary_}"
    name="${name%%.json}"
      finetuned+=("${name}")
 fi
done

# finetune with a checkpoint
for ckpt in out/checkpoints/*; do
# skip if already finetuned
   if [[ " ${finetuned[@]} " =~ " ${ckpt##*/} " ]]; then
      echo "Already finetuned ${ckpt##*/}"
      continue
   fi
   is_finished=$(ls ${ckpt} | grep "epoch_99.pt")
   if [[ -z $is_finished ]]; then
      echo "Not finished ${ckpt##*/}"
      continue
   fi
 echo "Finetuning ${ckpt##*/}"
#  python finetune_gt.py --ckpt ${ckpt##*/} --gpu_id 1
if [[ "${ckpt##*/}" =~ ^pretrain_gin.* ]]; then
   python finetune_gin.py --ckpt ${ckpt##*/} --gpu_id 1
   
else
   python finetune_gt.py --ckpt ${ckpt##*/} --gpu_id 1
 
fi
done



# no_pretrain
python finetune_gt.py --ckpt no_pretrain --gpu_id 1
python finetune_gin.py --ckpt no_pretrain --gpu_id 1