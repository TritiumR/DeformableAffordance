#!/bin/bash
for ((i=0;i<=99;i++))
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
  --exp_name 1111_15-S-no_unet-step4-1500-180000 \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-15-tryseven-S-no-unet-step4-step-4/critic-ckpt-180000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1111-06-online-no_unet-aff-step-5-0.06-step-1/attention-online-ckpt-1500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
