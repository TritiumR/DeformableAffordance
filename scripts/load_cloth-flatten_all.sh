#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=6 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1108_09-step5-both-online-5500 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9001-1107-02-2:1-both-online-step-5-aff-0.65-step-1/critic-online-ckpt-5500.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-9001-1107-02-2:1-both-online-step-5-aff-0.65-step-1/attention-online-ckpt-5500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
