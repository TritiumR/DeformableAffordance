CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 4000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1008-11-tryfive-U \
  --suffix tryfive-step1-U \
  --max_load 2000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --no_perturb \

