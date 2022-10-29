#!/bin/bash
for ((i=3;i<=4;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --shape S \
  --path ./data/rope-configuration-tryseven-step3-S \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000 - 12000)) \
  --step 3 \
  --headless 1 &
  echo "running $i"
done
