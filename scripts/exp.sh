#!/bin/bash
for ((i=0;i<=0;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step3-S \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 50000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1110-03-tryseven-S-no-global-step3-no_perturb \
  --max_load 5000 \
  --batch 20 \
  --unet 1 \
  --without_global \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1109-06-tryseven-no-global-aff-step2-step-1/attention-ckpt-300000.h5 \
  --no_perturb
  CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1110-04-tryseven-no-global-aff-step3 \
  --suffix tryseven-step3-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 1 \
  --without_global \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-03-tryseven-S-no-global-step3-no_perturb-step-3/critic-ckpt-50000.h5
done
