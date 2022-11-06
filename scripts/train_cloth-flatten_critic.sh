CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step3 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1106-24-tryeleven-no_unet-step3 \
  --max_load 5000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --unet 0 \
  --use_mask \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1106-02-2:1-no_unet-online-step-3-aff-0.65-step-1/attention-online-ckpt-7000.h5 \

