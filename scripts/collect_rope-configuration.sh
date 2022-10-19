#!/bin/bash
for ((i=4;i<=5;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --shape S \
  --path ./data/rope-configuration-tryseven-step1-S \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000 - 16000)) \
  --step 1 \
  --headless 1 &
  echo "running $i"
done
