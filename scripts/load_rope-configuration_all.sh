#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=2 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 2 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1030-11-S-aff-400000-critic-only_gt-step2-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1024-14-tryseven-S-step2-only_gt-mix1-no_perturb-step-2/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1029-04-tryseven-no_perturb-aff-S-step2-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 0 \
  --headless 1
done
