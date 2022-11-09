#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1109_19-S-step2-no_global \
  --test_id "$i" \
  --unet 1 \
  --without_global \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1107-22-tryseven-S-no_global-step2-no_perturb-step-2/critic-ckpt-200000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1109-06-tryseven-no-global-aff-step2-step-1/attention-ckpt-150000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
