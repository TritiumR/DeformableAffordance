#!/bin/bash
for ((i=0;i<=0;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --shape S \
  --path ./data/rope-configuration-trythree-step1-S \
  --process_num 1 \
  --data_num 1 \
  --data_type 10 \
  --curr_data $(($i * 1000)) \
  --step 1 \
  --headless 1
  echo "running $i"
done
