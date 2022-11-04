CUDA_VISIBLE_DEVICES=5 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 5 \
  --out_logits 1 \
  --exp_name 1104-03-2:1-mix1-online-step-5-aff-0.6 \
  --mode aff \
  --process_num 1 \
  --unet 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1102-01-tryeleven-step5-continue-step-5/critic-ckpt-100000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-08-tryeleven-aff-with-2:1-mix1-step5-step-1/attention-ckpt-400000.h5 \
  --image_size 160 \
  --headless 1

