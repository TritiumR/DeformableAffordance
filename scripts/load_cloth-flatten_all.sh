#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1101-09-step1-without_global-test-set_flat-10 \
  --test_id "$i" \
  --without_global \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-17-tryten-without_global-step-1/critic-ckpt-210000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1028-03-online-step-5-aff-0.6-step-1/attention-online-ckpt-9500.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
