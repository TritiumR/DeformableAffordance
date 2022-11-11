CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1111-07-tryeleven-no_unet-aff-step5-with-100000 \
  --suffix tryeleven-step5 \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --unet 0 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1110-12-tryeleven-no_unet-step5-step-5/critic-ckpt-100000.h5 \

