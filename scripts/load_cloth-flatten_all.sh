#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1106_23-step2-no-unet-online-aff_critic-set_flat-10 \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-09-tryeleven-no_unet-step2-step-2/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1106-02-2:1-no_unet-online-step-3-aff-0.65-step-1/attention-online-ckpt-7000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
