CUDA_VISIBLE_DEVICES=7 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7000 \
  --step 1 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 0920-7-step-1-all_score-critic-10 \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0907-02-tryseven-not_on_cloth_zero-step-1/critic-ckpt-300000.h5 \
  --critic_pick \
  --save_video_dir './test_video/' \
  --headless 1

