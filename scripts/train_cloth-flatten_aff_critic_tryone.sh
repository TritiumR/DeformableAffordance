python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 2500 \
  --num_iters 50000 \
  --out_logits 1 \
  --demo_times 5 \
  --exp_name 0813-01-trytwo \
  --suffix trytwo \
  --max_load 1000 \
  --batch 4 \

