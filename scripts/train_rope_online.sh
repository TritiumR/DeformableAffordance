CUDA_VISIBLE_DEVICES=6 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 8001 \
  --data_type 3 \
  --critic_type 1 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1105-04-S-2:1-mix2-online-aff-step-4-0.065 \
  --mode aff \
  --process_num 1 \
  --learning_rate 1e-4 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1101-05-tryseven-S-step3-2:1-mix2-no_perturb-step-3/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1104-04-tryseven-no_perturb-aff-with-2:1-mix2-S-step3-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --headless 1

