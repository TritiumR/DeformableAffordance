CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1108-08-tryeleven-no_global-aff-step3 \
  --suffix tryeleven-step3 \
  --max_load 3000 \
  --batch 20 \
  --model aff \
  --unet 1 \
  --without_global \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1107-11-tryeleven-no_global-step3-step-3/critic-ckpt-120000.h5 \

