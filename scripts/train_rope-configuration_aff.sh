CUDA_VISIBLE_DEVICES=6 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1102-03-tryseven-no_perturb-aff-with-only_gt-S-step3 \
  --suffix tryseven-step3-S \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1030-11-tryseven-S-step3-only_gt-no_perturb-step-3/critic-ckpt-300000.h5 \

