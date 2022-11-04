CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step2 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1103-09-tryeleven-no_unet-step2 \
  --max_load 3000 \
  --batch 20 \
  --learning_rate 5e-5 \
  --model critic \
  --image_size 160 \
  --unet 0 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1102-04-no_unet-online-step-2-aff-0.8-step-1/attention-online-ckpt-7000.h5 \

