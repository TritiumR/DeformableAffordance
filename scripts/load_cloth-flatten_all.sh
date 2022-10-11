CUDA_VISIBLE_DEVICES=6 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 10000 \
  --step 2 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1011-10-only_gt-step-2-all_score-10-critic \
  --suffix trynine \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1001-08-trynine-only_gt-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1001-08-trynine-only_gt-step-2 \
  --critic_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

