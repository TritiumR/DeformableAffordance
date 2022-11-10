#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=4 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1110_14-S-step3-no_unet-4500-200000 \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1108-01-tryseven-S-no_unet-step3-no_perturb-step-3/critic-ckpt-200000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1110-07-online-no_unet-aff-step-4-0.06-step-1/attention-online-ckpt-4500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
