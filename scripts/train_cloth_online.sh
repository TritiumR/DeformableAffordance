CUDA_VISIBLE_DEVICES=5 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 7001 \
  --data_type 3 \
  --critic_type 1 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1106-02-2:1-no_unet-online-step-3-aff-0.65 \
  --mode aff \
  --process_num 1 \
  --unet 0 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-09-tryeleven-no_unet-step2-step-2/critic-ckpt-270000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1105-05-tryeleven-no_unet-aff-step-2/attention-ckpt-300000.h5 \
  --image_size 160 \
  --headless 1

