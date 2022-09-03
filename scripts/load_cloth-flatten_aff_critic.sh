CUDA_VISIBLE_DEVICES=1 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 10300 \
  --out_logits 1 \
  --exp_name 0903-01-tryfour-expert \
  --suffix tryfour \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-4000-tryfive-step-1/critic-ckpt-200000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

