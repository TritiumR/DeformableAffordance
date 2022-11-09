CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1109-24-tryeleven-no_global-aff-step4 \
  --suffix tryeleven-step4 \
  --max_load 5000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --without_global \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1109-23-tryeleven-no_global-step4-step-4/critic-ckpt-100000.h5 \

