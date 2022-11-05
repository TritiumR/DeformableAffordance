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
  --exp_name 1105_23-S-no-online-step3-set_flat-trick-0.040 \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1030-11-tryseven-S-step3-only_gt-no_perturb-step-3/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1102-03-tryseven-no_perturb-aff-with-only_gt-S-step3-step-1/attention-ckpt-300000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
