CUDA_VISIBLE_DEVICES=4 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step2 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1011-04-tryten_eleven-2:1 \
  --max_load 2000 \
  --batch 32 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1004-01-tryten-aff-step-1/attention-ckpt-200000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1004-01-tryten-aff-step-1 \
  --model critic \
  --image_size 160 \

