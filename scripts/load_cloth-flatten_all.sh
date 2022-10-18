#!/bin/bash
for ((i=0;i<=19;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=1 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --step 2 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1018-02-step-2-online-both-4000-all_score-aff_critic \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1/critic-online-ckpt-4000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1/attention-online-ckpt-4000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1 \
  --image_size 160 \
  --set_flat 0 \
  --save_video_dir './test_video/' \
  --headless 1
done
