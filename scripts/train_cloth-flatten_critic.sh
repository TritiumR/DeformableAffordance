CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step4 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 4 \
  --exp_name 1023-03-tryeleven-only_gt-step4 \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --only_gt \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \

