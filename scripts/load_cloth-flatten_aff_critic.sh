CUDA_VISIBLE_DEVICES=7 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 10000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1003-08-step-2-60000-expert \
  --suffix trynine \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1001-07-trynine-step-2/critic-ckpt-60000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1001-07-trynine-step-2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-6000-0930-06-trynine-aff-step-1/attention-ckpt-200000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-6000-0930-06-trynine-aff-step-1/ \
  --expert_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

