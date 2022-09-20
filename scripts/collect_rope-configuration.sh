#!/bin/bash
for ((i=5;i<=7;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --path ./data/rope-configuration-trytwo-step2-S \
  --process_num 2 \
  --data_num 1000 \
  --data_type 10 \
  --curr_data $(($i * 1000)) \
  --step 2 \
  --headless 1
  echo "running $i"
done
