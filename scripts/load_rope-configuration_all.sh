#!/bin/bash
for ((i=0;i<=19;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=1 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape U \
  --agent aff_critic \
  --num_demos 7900 \
  --step 1 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1022-06-U-300000-step1-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1021-13-trysix-no_perturb-aff-U-step-1/attention-ckpt-360000.h5 \
  --load_aff_mean_std_dir checkpoints/rope-configuration-Aff_Critic-8000-1021-13-trysix-no_perturb-aff-U-step-1 \
  --image_size 160 \
  --set_flat 0 \
  --headless 1
done
