CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1110-02-tryseven-no-global-aff-step3 \
  --suffix tryseven-step3-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 1 \
  --without_global \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-03-tryseven-S-no-global-step3-no_perturb-step-3/critic-ckpt-100000.h5

