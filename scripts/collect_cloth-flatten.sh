#!/bin/bash
for ((i=0;i<=6;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryseven-step2 \
  --process_num 2 \
  --data_num 500 \
  --data_type 5 \
  --curr_data $(($i * 1000)) \
  --step 2 \
  --headless 1 &\
  echo "running $i"
done
