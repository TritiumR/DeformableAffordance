#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 7900 \
  --step 1 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1024-06-S-online-aff-9500-critic-300000-step1-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1/attention-online-ckpt-9500.h5 \
  --load_aff_mean_std_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1 \
  --image_size 160 \
  --set_flat 0 \
  --headless 1
done
