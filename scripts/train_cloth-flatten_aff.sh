CUDA_VISIBLE_DEVICES=6 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1106-08-tryeleven-no_global-aff \
  --suffix tryeleven-step2 \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --without_global \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-09-tryeleven-without_global-step2-step-2/critic-ckpt-210000.h5 \

