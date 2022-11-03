CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1103-08-tryeleven-aff-with-2:1-mix1-step5 \
  --suffix tryeleven-step5 \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1102-01-tryeleven-step5-continue-step-5/critic-ckpt-100000.h5 \

