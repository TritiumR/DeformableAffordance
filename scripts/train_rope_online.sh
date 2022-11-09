CUDA_VISIBLE_DEVICES=1 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 9001 \
  --data_type 3 \
  --critic_type 1 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1109-02-online-aff-step-5-0.06 \
  --mode aff \
  --process_num 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1107-03-tryseven-S-step4-2:1-mix3-no_perturb-step-4/critic-ckpt-180000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1109-01-tryseven-with-180000-aff-step4-step-1/attention-ckpt-300000.h5 \
  --image_size 160 \
  --headless 1

