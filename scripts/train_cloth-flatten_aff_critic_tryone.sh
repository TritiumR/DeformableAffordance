python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 1 \
  --num_iters 10000 \
  --out_logits 1 \
  --demo_times 1 \
  --exp_name 0809-01-tryone \
  --suffix tryone \
  --max_load 1000 \
  --batch 1 \

