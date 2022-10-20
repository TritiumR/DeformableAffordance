CUDA_VISIBLE_DEVICES=7 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1020-04-both-online-2500-step-3 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/critic-online-ckpt-2500.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-2500.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

