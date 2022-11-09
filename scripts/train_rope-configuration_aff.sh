CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1109-06-tryseven-no-global-aff-step2 \
  --suffix tryseven-step2-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 1 \
  --without_global \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1107-22-tryseven-S-no_global-step2-no_perturb-step-2/critic-ckpt-200000.h5 \

