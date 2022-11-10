#!/bin/bash
for ((i=0;i<=0;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step4-S \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 200000 \
  --out_logits 1 \
  --step 4 \
  --exp_name 1110-15-tryseven-S-no-unet-step4 \
  --max_load 5000 \
  --batch 20 \
  --unet 0 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-7001-1110-07-online-no_unet-aff-step-4-0.06-step-1/attention-online-ckpt-4500.h5 \
  --no_perturb
  CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1110-13-tryeleven-no_unet-aff-step5 \
  --suffix tryeleven-step5 \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --unet 0 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1110-12-tryeleven-no_unet-step5-step-5/critic-ckpt-200000.h5
done
