CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step3 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1107-11-tryeleven-no_global-step3 \
  --max_load 5000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --unet 1 \
  --without_global \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1106-26-2:1-no_global-online-step-3-aff-0.7-step-1/attention-online-ckpt-5500.h5 \

