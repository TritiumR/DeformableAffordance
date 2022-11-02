CUDA_VISIBLE_DEVICES=2 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1102-04-no_unet-online-step-2-aff-0.8 \
  --mode aff \
  --process_num 1 \
  --unet 0 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-16-tryten-no_unet-step-1/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1101-11-tryeleven-aff-no_unet-step1-step-1/attention-ckpt-360000.h5 \
  --image_size 160 \
  --headless 1

