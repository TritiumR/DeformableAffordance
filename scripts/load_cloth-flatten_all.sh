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
  --exp_name 1030-17-step4-online-continue-critic-set_flat-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1029-08-tryeleven-2:1-mix1-step-4/critic-ckpt-100000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1028-03-online-step-5-aff-0.6-step-1/attention-online-ckpt-9500.h5 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 1 \
  --headless 1
done
