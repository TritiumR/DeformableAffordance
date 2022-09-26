CUDA_VISIBLE_DEVICES=1 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --step 1 \
  --out_logits 1 \
  --exp_name 0925-04-step-1-test \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0907-02-tryseven-not_on_cloth_zero-step-1/critic-ckpt-300000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

