CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1019-02-eleven-both_online-4000-step-2 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1018-08-online-step-2-both-0.8-step-1/critic-online-ckpt-4000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1018-08-online-step-2-both-0.8-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1018-08-online-step-2-both-0.8-step-1/attention-online-ckpt-4000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1018-08-online-step-2-both-0.8-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

