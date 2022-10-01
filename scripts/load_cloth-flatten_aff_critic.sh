CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1001-05-step-1-106000-200000-critic \
  --suffix trynine \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-6000-0928-01-trynine-step-1/critic-ckpt-106000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-6000-0930-06-trynine-aff-step-1/attention-ckpt-200000.h5 \
  --load_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-6000-0930-06-trynine-aff-step-1 \
  --critic_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

