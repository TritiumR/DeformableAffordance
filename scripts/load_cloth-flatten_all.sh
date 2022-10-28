#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=6 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 12000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1028-06-12000-step3-online-set_flat-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-12000-1026-07-tryeleven-2:1-mix2-step-3/critic-ckpt-150000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-12000-1026-07-tryeleven-2:1-mix2-step-3 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-7500.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 1 \
  --headless 1
done
