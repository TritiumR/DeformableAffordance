CUDA_VISIBLE_DEVICES=2 python ./softgym/test_all.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --step 2 \
  --test_step 10 \
  --out_logits 1 \
  --exp_name 1014-02-step-2-only_gt-all_score-10-critic \
  --suffix tryeleven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1012-02-tryten_eleven-only_gt-step-2/critic-ckpt-225000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1012-02-tryten_eleven-only_gt-step-2 \
  --critic_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

