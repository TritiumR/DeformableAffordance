CUDA_VISIBLE_DEVICES=1 python ./softgym/train_online.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --shape S \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 5 \
  --out_logits 1 \
  --exp_name 1111-06-online-no_unet-aff-step-5-0.06 \
  --mode aff \
  --unet 0 \
  --process_num 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-15-tryseven-S-no-unet-step4-step-4/critic-ckpt-140000.h5 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-8000-1111-01-tryseven-no-unet-aff-step4-with-60000-step-1/attention-ckpt-200000.h5 \
  --image_size 160 \
  --headless 1

