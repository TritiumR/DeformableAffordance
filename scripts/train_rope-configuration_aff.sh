CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 5000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0923-02-trytwo-aff-not_on_cloth_zero-U \
  --suffix tryone-step1-U \
  --max_load 2000 \
  --batch 10 \
  --learning_rate 1e-4 \
  --model aff \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-5000-0919-02-trytwo-step1-U-step-1/critic-ckpt-300000.h5 \

