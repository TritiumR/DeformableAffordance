CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step2 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1103-09-tryeleven-without_global-step2 \
  --max_load 5000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --unet 1 \
  --without_global \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7001-1103-02-no_global-online-step-2-aff-0.8-step-1/attention-online-ckpt-7000.h5 \

