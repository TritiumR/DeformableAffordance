#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --test_step 10 \
  --exp_name rope-configuration-test \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-placing-step-4/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-IST-step-4/attention-online-ckpt-7000.h5 \
  --headless 1 \
  --save_video_dir ./test_video
done
