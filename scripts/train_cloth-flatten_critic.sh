CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step4 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 100000 \
  --out_logits 1 \
  --step 4 \
  --exp_name 1110-10-tryeleven-no_unet-step4-continue \
  --max_load 5000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --unet 0 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1109-05-no_unet-online-step-4-aff-step-1/attention-online-ckpt-7000.h5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1107-11-tryeleven-no_unet-step4-step-4/critic-ckpt-100000.h5 \

