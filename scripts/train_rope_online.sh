CUDA_VISIBLE_DEVICES=7 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape U \
  --num_online 10000 \
  --data_type 4 \
  --critic_type 1 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1022-04-online-aff-step-2-0.06 \
  --mode aff \
  --process_num 1 \
  --critic_depth 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1021-13-trysix-no_perturb-aff-U-step-1/attention-ckpt-360000.h5 \
  --load_aff_mean_std_dir checkpoints/rope-configuration-Aff_Critic-8000-1021-13-trysix-no_perturb-aff-U-step-1 \
  --image_size 160 \
  --headless 1

