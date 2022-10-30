CUDA_VISIBLE_DEVICES=1 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 10000 \
  --data_type 4 \
  --critic_type 1 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1030-05-S-online-aff-step-3-0.065 \
  --mode aff \
  --process_num 1 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1026-18-tryseven-S-step2-with-online-7000-2:1-no_perturb-step-2/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1029-04-tryseven-no_perturb-aff-S-step2-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --headless 1

