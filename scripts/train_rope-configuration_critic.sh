CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 5000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --demo_times 10 \
  --exp_name 0919-02-trytwo-step1-U \
  --suffix tryone-step1-U \
  --max_load 2000 \
  --batch 4 \
  --model critic \
  --no_perturb

