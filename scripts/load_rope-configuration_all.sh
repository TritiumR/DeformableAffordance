#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1031-03-S-online-aff-9500-critic-step2-set_flat \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1026-18-tryseven-S-step2-with-online-7000-2:1-no_perturb-step-2/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-10000-1030-05-S-online-aff-step-3-0.065-step-1/attention-online-ckpt-9500.h5 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 1 \
  --headless 1
done
