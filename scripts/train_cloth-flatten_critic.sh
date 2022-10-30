CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryten-step1 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --without_global \
  --step 1 \
  --exp_name 1030-17-tryten-without_global \
  --max_load 3000 \
  --batch 20 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \

