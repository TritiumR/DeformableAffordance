#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=6 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1109_18-S-step4-random-pick \
  --test_id "$i" \
  --unet 1 \
  --random_pick \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1107-03-tryseven-S-step4-2:1-mix3-no_perturb-step-4/critic-ckpt-240000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-9001-1109-02-online-aff-step-5-0.06-step-1/attention-online-ckpt-500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
