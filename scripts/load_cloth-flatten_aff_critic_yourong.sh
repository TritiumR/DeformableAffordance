CUDA_VISIBLE_DEVICES=4 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1007-01-90000-step-1-no_perturb-slower \
  --suffix tryten \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1006-05-tryten-no_perturb-step-1/critic-ckpt-90000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1006-05-tryten-no_perturb-step-1 \
  --expert_pick \
  --image_size 160 \
  --save_video_dir './test_video/' \
  --headless 1

