#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=1 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1105-06-S-step1-set_flat-trick \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1/attention-online-ckpt-9000.h5 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 1 \
  --headless 1
done
