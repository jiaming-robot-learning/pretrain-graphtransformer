

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


if [[ "${ckpt##*/}" =~ ^pretrain_gin.* ]]; then
   python finetune_gin.py --ckpt ${ckpt##*/} --gpu_id 3
   
else
   arrin=(${ckpt//_/ })
   finetunecfg="finetune_${arrin[1]}".json
   if [ -f config/$finetunecfg ]; then
      echo " Using config file $finetunecfg"
   else
      echo " Using default config file finetune.json"
      finetunecfg="finetune.json"
   fi
   python finetune_gt.py --ckpt ${ckpt##*/} --gpu_id 3 --config config/$finetunecfg
fi
done



# # no_pretrain
# python finetune_gt.py --ckpt no_pretrain --gpu_id 1
# python finetune_gin.py --ckpt no_pretrain --gpu_id 1


# #  gt variants no pretrain
# python finetune_gt.py --ckpt no_pretrain_gtnope --gpu_id 1 --config config/finetune_gtnope.json
# python finetune_gt.py --ckpt no_pretrain_gtrwpe --gpu_id 1 --config config/finetune_gtrwpe.json
# python finetune_gt.py --ckpt no_pretrain_gtsmall --gpu_id 1 --config config/finetune_gtsmall.json
# python finetune_gt.py --ckpt no_pretrain_gtnopesmall --gpu_id 1 --config config/finetune_gtnopesmall.json
# python finetune_gt.py --ckpt no_pretrain_gtnopesmaller --gpu_id 1 --config config/finetune_gtnopesmaller.json

python finetune_gt.py --ckpt no_pretrain_gtrwpesmall --gpu_id 1 --config config/finetune_gtrwpesmall.json