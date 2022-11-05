#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1105-01-only_gt-aff_critic-set_flat-10 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1101-04-tryeleven-only_gt-step-5/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-9000-1104-02-tryeleven-only_gt-aff-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
