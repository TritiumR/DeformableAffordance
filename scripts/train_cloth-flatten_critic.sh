CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryten-step1 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --exp_name 1008-06-tryten-new_preprocess \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 3e-4 \
  --model critic \
  --image_size 160 \

