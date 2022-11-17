#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=2 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1117_04-S-unet-no-online-step5-7000-300000 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1112-02-tryseven-S-step5-mix4-step-5/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1115-02-online-unet-aff-step-6-0.06-step-1/attention-online-ckpt-0.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
