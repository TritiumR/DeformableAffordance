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
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1107_19-S-step3-both-online-9000 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-9001-1106-25-S-2:1-online-both-step-4-0.06-step-1/critic-online-ckpt-9000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-9001-1106-25-S-2:1-online-both-step-4-0.06-step-1/attention-online-ckpt-9000.h5\
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
