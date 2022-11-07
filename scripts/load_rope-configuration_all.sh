#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1107_07-S-random_pick-step2-set_flat \
  --test_id "$i" \
  --unet 1 \
  --random_pick \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1026-18-tryseven-S-step2-with-online-7000-2:1-no_perturb-step-2/critic-ckpt-300000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
