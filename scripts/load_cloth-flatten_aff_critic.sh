CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1019-05-eleven-300000-step-3 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1016-04-tryeleven-online_step3-step-3/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1016-04-tryeleven-online_step3-step-3 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1018-06-tryeleven-aff-step3-step-1/attention-ckpt-400000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1018-06-tryeleven-aff-step3-step-1 \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

