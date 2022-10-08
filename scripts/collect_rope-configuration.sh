#!/bin/bash
for ((i=0;i<=0;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name RopeConfiguration \
  --shape U \
  --path ./data/rope-configuration-trysix-step1-U \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000)) \
  --step 1 \
  --headless 1
  echo "running $i"
done
