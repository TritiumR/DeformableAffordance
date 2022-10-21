CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1021-13-trysix-no_perturb-aff-U \
  --suffix trysix-step1-U \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/rope-configuration-Aff_Critic-8000-1019-02-trysix-U-step-1 \

