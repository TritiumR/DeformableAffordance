CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1110-01-tryeleven-no_unet-aff-step4 \
  --suffix tryeleven-step4 \
  --max_load 5000 \
  --batch 20 \
  --model aff \
  --unet 0 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1107-11-tryeleven-no_unet-step4-step-4/critic-ckpt-70000.h5 \

