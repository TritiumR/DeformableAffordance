CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1008-09-tryten-aff-only_depth \
  --suffix tryten-step1 \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --only_depth \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1/critic-ckpt-301000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1 \

