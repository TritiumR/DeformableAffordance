CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 6000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --demo_times 10 \
  --exp_name 0917-06-trytwo-step1-S \
  --suffix tryone-step1-S \
  --max_load 3000 \
  --batch 4 \
  --model critic \

