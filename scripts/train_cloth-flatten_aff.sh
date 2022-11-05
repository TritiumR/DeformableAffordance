CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1105-05-tryeleven-no_unet-aff \
  --suffix tryeleven-step2 \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 0 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1103-09-tryeleven-no_unet-step2-step-2/critic-ckpt-210000.h5 \

