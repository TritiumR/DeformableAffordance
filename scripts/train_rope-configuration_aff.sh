CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1029-04-tryseven-no_perturb-aff-S-step2 \
  --suffix tryseven-step2-S \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1026-18-tryseven-S-step2-with-online-7000-2:1-no_perturb-step-2/critic-ckpt-210000.h5 \

