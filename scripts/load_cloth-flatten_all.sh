#!/bin/bash
for ((i=0;i<=29;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1028-04-step4-no_online-set_flat-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4/critic-ckpt-195000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1026-22-tryeleven-aff-with-2:1-195000-step4-step-1/attention-ckpt-400000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1026-22-tryeleven-aff-with-2:1-195000-step4-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --set_flat 1 \
  --headless 1
done
