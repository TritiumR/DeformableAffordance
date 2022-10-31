CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1031-04-tryseven-no_perturb-aff-with-only_gt-S-step2 \
  --suffix tryseven-step2-S \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1024-14-tryseven-S-step2-only_gt-mix1-no_perturb-step-2/critic-ckpt-300000.h5 \

