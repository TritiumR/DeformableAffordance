#!/bin/bash
for ((i=2;i<=7;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryseven-step1 \
  --process_num 2 \
  --data_num 1000 \
  --data_type 5 \
  --curr_data $(($i * 1000 - 2000)) \
  --step 1 \
  --headless 1 &\
  echo "running $i"
done
