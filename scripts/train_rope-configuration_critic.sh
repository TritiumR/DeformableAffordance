CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step2-S \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1026-16-tryseven-S-step2-only_state-no_perturb \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --no_perturb \
  --only_state \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/rope-configuration-Aff_Critic-10000-1023-06-S-online-aff-step-2-0.06-step-1 \
