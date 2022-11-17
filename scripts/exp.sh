#!/bin/bash
for ((i=0;i<=0;i++))
do
  echo "running critic"
  CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1115-03-tryseven-no-unet-step5-aff \
  --suffix tryseven-step5-S \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1114-01-tryseven-S-step5-no_unet-step-5/critic-ckpt-240000.h5
  echo "running aff"
  CUDA_VISIBLE_DEVICES=3 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 6 \
  --out_logits 1 \
  --exp_name 1115-04-online-no-unet-aff-step-6-0.06 \
  --mode aff \
  --unet 0 \
  --process_num 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1114-01-tryseven-S-step5-no_unet-step-5/critic-ckpt-240000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1115-03-tryseven-no-unet-step5-aff-step-1/attention-ckpt-300000.h5 \
  --image_size 160 \
  --headless 1
done
