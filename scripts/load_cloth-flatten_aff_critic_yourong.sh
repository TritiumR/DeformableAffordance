CUDA_VISIBLE_DEVICES=7 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1009-05-40000-ten-step-1 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1/critic-ckpt-301000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1005-01-tryten-aff-lr-step-1/attention-ckpt-200000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1005-01-tryten-aff-lr-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

