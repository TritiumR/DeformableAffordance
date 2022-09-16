CUDA_VISIBLE_DEVICES=0,1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --demo_times 10 \
  --exp_name 0915-04-trymulti \
  --suffix tryseven \
  --max_load 3000 \
  --batch 10 \
  --learning_rate 5e-4 \
  --model critic \
  --multi_gpu \

