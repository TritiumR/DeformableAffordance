CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 5 \
  --out_logits 1 \
  --exp_name 1028-07-critic-180000-only_gt-step-5 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --expert_pick \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1026-06-tryeleven-only_gt-step-5/critic-ckpt-180000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1026-06-tryeleven-only_gt-step-5 \
  --image_size 160 \
  --headless 1

