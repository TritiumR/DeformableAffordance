CUDA_VISIBLE_DEVICES=6 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1106-26-2:1-no_global-online-step-3-aff-0.7 \
  --mode aff \
  --process_num 1 \
  --unet 1 \
  --learning_rate 5e-5 \
  --without_global \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-09-tryeleven-without_global-step2-step-2/critic-ckpt-270000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1106-08-tryeleven-no_global-aff-step-1/attention-ckpt-300000.h5 \
  --image_size 160 \
  --headless 1

