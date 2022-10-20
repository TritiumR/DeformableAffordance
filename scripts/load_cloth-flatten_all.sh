#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=3 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 5 \
  --out_logits 1 \
  --exp_name 1020-13-both-online-3500-step3-5 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/critic-online-ckpt-3500.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-3500.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --image_size 160 \
  --set_flat 0 \
  --save_video_dir './test_video/' \
  --headless 1
done
