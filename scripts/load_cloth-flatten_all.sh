#!/bin/bash
for ((i=0;i<=19;i++))
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
  --exp_name 1019-07-only_gt-step-3-no_online \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1017-02-tryeleven-online-only_gt-step3-step-3/critic-ckpt-210000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1017-02-tryeleven-online-only_gt-step3-step-3 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1018-06-tryeleven-aff-step3-step-1/attention-ckpt-400000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1018-06-tryeleven-aff-step3-step-1 \
  --image_size 160 \
  --set_flat 0 \
  --save_video_dir './test_video/' \
  --headless 1
done
