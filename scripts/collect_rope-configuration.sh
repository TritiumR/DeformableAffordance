#!/bin/bash
for ((i=7;i<=7;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --shape S \
  --path ./data/rope-configuration-tryseven-step2-S \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000 - 24000)) \
  --step 2 \
  --headless 1 &
  echo "running $i"
done
