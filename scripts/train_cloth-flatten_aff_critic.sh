CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 2500 \
  --num_iters 100000 \
  --out_logits 1 \
  --demo_times 5 \
  --exp_name 0813-01-trythree \
  --suffix trytwo \
  --max_load 1000 \
  --batch 4 \

