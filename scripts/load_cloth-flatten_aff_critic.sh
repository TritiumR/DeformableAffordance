CUDA_VISIBLE_DEVICES=3 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 10000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1009-09-ten-400000-step-2-120000-expert \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1006-02-tryten-400000-aff-lr_5e-5-step-2/critic-ckpt-120000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1006-02-tryten-400000-aff-lr_5e-5-step-2 \
  --expert_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

