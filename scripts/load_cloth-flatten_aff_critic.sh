CUDA_VISIBLE_DEVICES=4 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 10000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1008-10-nine_ten-60000-step-2-exp \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1004-07-trynine_ten-step-2/critic-ckpt-60000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1004-07-trynine_ten-step-2 \
  --image_size 160 \
  --exp \
  --save_video_dir './test_video/' \
  --headless 1

