CUDA_VISIBLE_DEVICES=1 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1110-07-online-no_unet-aff-step-4-0.06 \
  --mode aff \
  --unet 0 \
  --process_num 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1108-01-tryseven-S-no_unet-step3-no_perturb-step-3/critic-ckpt-200000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-02-tryseven-no-unet-aff-step3-step-1/attention-ckpt-140000.h5 \
  --image_size 160 \
  --headless 1

