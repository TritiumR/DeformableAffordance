#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1109_16-step3-no-unet-1500-270000 \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1106-24-tryeleven-no_unet-step3-step-3/critic-ckpt-270000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1109-05-no_unet-online-step-4-aff-step-1/attention-online-ckpt-1500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
