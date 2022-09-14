#!/bin/bash
for ((i=2;i<=7;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --path ./data/rope-configuration-tryone-step1-S \
  --process_num 2 \
  --data_num 500 \
  --data_type 10 \
  --curr_data $(($i * 1000)) \
  --step 1 \
  --headless 1 &
  echo "running $i"
done
