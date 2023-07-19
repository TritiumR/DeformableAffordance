#!/bin/bash
for ((i=1;i<=5;i++))
do
  echo "running step $i placing"
  if [ 1 = "$i" ]
  then
    CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
    --env_name ClothFlatten \
    --task cloth-flatten \
    --suffix step-"$i" \
    --num_demos 8000 \
    --demo_times 10 \
    --num_iters 300000 \
    --step "$i" \
    --exp_name placing \
    --max_load 5000 \
    --batch 20 \
    --learning_rate 1e-4 \
    --model critic
  else
    CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
    --env_name ClothFlatten \
    --task cloth-flatten \
    --suffix step-"$i" \
    --num_demos 8000 \
    --demo_times 10 \
    --num_iters 300000 \
    --step "$i" \
    --exp_name placing \
    --max_load 5000 \
    --batch 20 \
    --learning_rate 1e-4 \
    --model critic \
    --load_aff_dir checkpoints/cloth-flatten-IST-step-"$i"/attention-online-ckpt-7000.h5
  fi
  echo "running step $i picking"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --suffix step-"$i" \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --step "$i" \
  --exp_name picking \
  --max_load 5000 \
  --batch 20 \
  --model aff \
  --load_critic_dir checkpoints/cloth-flatten-placing-step-"$i"/critic-ckpt-300000.h5
  echo "running step $i IST"
  CUDA_VISIBLE_DEVICES=0 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --num_online 7001 \
  --step $((i + 1)) \
  --exp_name IST \
  --mode aff \
  --process_num 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-placing-step-"$i"/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-picking-step-"$i"/attention-ckpt-300000.h5 \
  --headless 1
done

