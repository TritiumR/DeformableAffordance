CUDA_VISIBLE_DEVICES=0 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 5 \
  --out_logits 1 \
  --exp_name 1110-08-no_unet-online-step-5-aff \
  --mode aff \
  --process_num 1 \
  --unet 0 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1107-11-tryeleven-no_unet-step4-step-4/critic-ckpt-100000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1110-01-tryeleven-no_unet-aff-step4-step-1/attention-ckpt-140000.h5 \
  --image_size 160 \
  --headless 1

