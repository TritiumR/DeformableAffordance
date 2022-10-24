CUDA_VISIBLE_DEVICES=4 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1024-01-270000-step-4-expert-1:1 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1021-15-tryeleven-online-5000-step4-1:1-step-3/critic-ckpt-270000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1021-15-tryeleven-online-5000-step4-1:1-step-3 \
  --expert_pick \
  --image_size 160 \
  --headless 1

