#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=3 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1103-01-step1-without_global-no_online-set_flat-10 \
  --test_id "$i" \
  --unet 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-17-tryten-without_global-step-1/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1102-05-tryeleven-aff-no_global-step1-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
