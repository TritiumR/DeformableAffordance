#!/bin/bash
for ((i=0;i<=59;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1021-10-only_gt-300000-step2-set_flat-10 \
  --test_id "$i" \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-15-tryten_eleven-online-only_state-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-15-tryten_eleven-online-only_state-step-2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1 \
  --image_size 160 \
  --set_flat 1 \
  --save_video_dir './test_video/' \
  --headless 1
done
