CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6300 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0823-01-tryfive \
  --suffix tryfour \
  --max_load 2000 \
  --batch 3 \

