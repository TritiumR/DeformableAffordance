#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=6 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1105_24-step5-no-online-aff_critic-set_flat-10 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1102-01-tryeleven-step5-continue-step-5/critic-ckpt-100000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1104-03-2:1-mix1-online-step-5-aff-0.6-step-1/attention-online-ckpt-0.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
