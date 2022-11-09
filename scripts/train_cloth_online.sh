CUDA_VISIBLE_DEVICES=1 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1109-05-no_unet-online-step-4-aff \
  --mode aff \
  --process_num 1 \
  --unet 0 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1106-24-tryeleven-no_unet-step3-step-3/critic-ckpt-270000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1108-07-tryeleven-no_unet-aff-step3-step-1/attention-ckpt-270000.h5 \
  --image_size 160 \
  --headless 1

