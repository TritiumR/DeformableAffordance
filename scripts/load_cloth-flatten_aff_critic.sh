CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --out_logits 1 \
  --exp_name 0909-01-tryseven-expert \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0907-02-tryseven-not_on_cloth_zero-step-1/critic-ckpt-300000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

