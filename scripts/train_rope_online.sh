CUDA_VISIBLE_DEVICES=5 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 9001 \
  --data_type 3 \
  --critic_type 5 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1106-25-S-2:1-online-both-step-4-0.06 \
  --mode both \
  --process_num 1 \
  --learning_rate 5e-5 \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1101-05-tryseven-S-step3-2:1-mix2-no_perturb-step-3/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1104-04-tryseven-no_perturb-aff-with-2:1-mix2-S-step3-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --headless 1

