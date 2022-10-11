CUDA_VISIBLE_DEVICES=6 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1011-10-ten-online-step-1 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1/critic-ckpt-301000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

