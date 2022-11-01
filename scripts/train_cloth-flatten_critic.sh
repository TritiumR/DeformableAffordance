CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-only_gt \
  --num_demos 9000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 5 \
  --exp_name 1101-04-tryeleven-only_gt \
  --only_gt \
  --max_load 3000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \

