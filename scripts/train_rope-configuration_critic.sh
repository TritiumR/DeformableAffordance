CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix trysix-step1-U \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --exp_name 1019-02-trysix-U \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --no_perturb \

