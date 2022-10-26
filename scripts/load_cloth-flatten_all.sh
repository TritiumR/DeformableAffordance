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
  --exp_name 1025-04-300000-step3-no_online-set_flat-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/critic-online-ckpt-7500.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-0.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
