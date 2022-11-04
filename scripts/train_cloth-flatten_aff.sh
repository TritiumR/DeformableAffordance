CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1104-02-tryeleven-only_gt-aff \
  --suffix tryeleven-only_gt \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1101-04-tryeleven-only_gt-step-5/critic-ckpt-300000.h5 \

