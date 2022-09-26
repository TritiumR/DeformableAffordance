#!/bin/bash
for ((i=1;i<=1;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-trynine-step1 \
  --process_num 2 \
  --data_num 3000 \
  --data_type 10 \
  --curr_data $(($i * 1000 - 1000)) \
  --step 1 \
  --headless 1 
  echo "running $i"
done
