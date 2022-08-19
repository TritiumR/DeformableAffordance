CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 3300 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0819-01-tryfour \
  --suffix tryfour \
  --max_load 1000 \
  --batch 3 \

