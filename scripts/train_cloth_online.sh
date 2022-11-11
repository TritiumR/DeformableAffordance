CUDA_VISIBLE_DEVICES=0 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 5 \
  --out_logits 1 \
  --exp_name 1111-17-no_unet-online-step-6-aff \
  --mode aff \
  --process_num 1 \
  --unet 0 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1110-12-tryeleven-no_unet-step5-step-5/critic-ckpt-140000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1111-07-tryeleven-no_unet-aff-step5-with-100000-step-1/attention-ckpt-140000.h5 \
  --image_size 160 \
  --headless 1

