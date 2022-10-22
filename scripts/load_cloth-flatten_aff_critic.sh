CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1023-02-120000-step-4-1:1 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1021-15-tryeleven-online-5000-step4-1:1-step-3/critic-ckpt-120000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1021-15-tryeleven-online-5000-step4-1:1-step-3 \
  --expert_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

