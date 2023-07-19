#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --test_step 10 \
  --exp_name cloth-flatten-test \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-placing-step-5/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-IST-step-5/attention-online-ckpt-7000.h5 \
  --headless 1 \
  --save_video_dir ./test_video
done
