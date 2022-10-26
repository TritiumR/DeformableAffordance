CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --shape S \
  --step 2 \
  --agent aff_critic \
  --num_demos 7900 \
  --out_logits 1 \
  --exp_name 1026-15-debug-step2 \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1/attention-online-ckpt-9500.h5 \
  --load_aff_mean_std_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1 \
  --exp \
  --image_size 160 \
  --headless 1
