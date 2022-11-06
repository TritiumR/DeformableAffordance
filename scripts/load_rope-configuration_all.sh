#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=1 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1106_11-S-no_unet-online-step1-no-set_flat-trick-0.040 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1101-05-tryseven-S-step3-2:1-mix2-no_perturb-step-3/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-9001-1105-17-S-2:1-mix2-online-aff-step-4-0.065-more-type-step-1/attention-online-ckpt-7500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
