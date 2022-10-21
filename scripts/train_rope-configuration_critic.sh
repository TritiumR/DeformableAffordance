CUDA_VISIBLE_DEVICES=6 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step1-S \
  --num_demos 7900 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --exp_name 1021-17-tryseven-S-perturb \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \

