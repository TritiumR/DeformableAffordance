CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 7900 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1026-12-tryseven-no_perturb-aff-S-test \
  --suffix tryseven-step1-S \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-7900-1021-16-tryseven-S-no_perturb-step-1 \

