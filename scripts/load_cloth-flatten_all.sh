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
  --exp_name 1111_16-step5-140000-140000-no-unet \
  --test_id "$i" \
  --unet 0 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1110-12-tryeleven-no_unet-step5-step-5/critic-ckpt-140000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1111-07-tryeleven-no_unet-aff-step5-with-100000-step-1/attention-ckpt-140000.h5 \
  --image_size 160 \
  --set_flat 1 \
  --headless 1
done
