CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1115-01-tryseven-step5-aff \
  --suffix tryseven-step5-S \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1112-02-tryseven-S-step5-mix4-step-5/critic-ckpt-300000.h5 \

