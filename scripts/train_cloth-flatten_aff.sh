CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1102-05-tryeleven-aff-no_global-step1 \
  --suffix tryten-step1 \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --without_global \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-17-tryten-without_global-step-1/critic-ckpt-300000.h5 \

