CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1109-01-tryseven-with-180000-aff-step4 \
  --suffix tryseven-step4-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 1 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1107-03-tryseven-S-step4-2:1-mix3-no_perturb-step-4/critic-ckpt-180000.h5 \

