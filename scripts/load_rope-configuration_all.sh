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
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1108_07-S-step2-no_unet \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1106-07-no_unet-tryseven-S-step2-no_perturb-step-2/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1108-03-no_unet-online-aff-step-3-0.06-step-1/attention-online-ckpt-7000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
