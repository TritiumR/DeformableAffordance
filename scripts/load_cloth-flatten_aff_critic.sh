CUDA_VISIBLE_DEVICES=7 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1017-03-ten_eleven-only_state-step-2-expert \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-15-tryten_eleven-online-only_state-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-15-tryten_eleven-online-only_state-step-2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1 \
  --expert_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

