#!/bin/bash
for ((i=1;i<=4;i++))
do
  echo "running $i"
  CUDA_VISIBLE_DEVICES=0 python softgym/collect.py \
  --env_name RopeConfiguration \
  --path ./data/rope-configuration-step-"$i" \
  --process_num 2 \
  --data_num 4000 \
  --data_type 10 \
  --curr_data 0 \
  --step $i \
  --headless 1
done
