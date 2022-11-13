#!/bin/bash
for ((i=0;i<=0;i++))
do
  echo "running critic"
  CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step5-S \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 5 \
  --exp_name 1114-01-tryseven-S-step5-no_unet \
  --max_load 8000 \
  --batch 20 \
  --unet 0 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1111-06-online-no_unet-aff-step-5-0.06-step-1/attention-online-ckpt-7000.h5 \
  --no_perturb
  echo "running aff"
  CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1114-02-tryseven-no_unet-step5-aff \
  --suffix tryseven-step5-S \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-1114-01-tryseven-S-step5-no_unet-step-5/critic-ckpt-300000.h5
done
