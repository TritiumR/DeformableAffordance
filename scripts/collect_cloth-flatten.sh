#!/bin/bash
for ((i=4;i<=4;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryten-step2 \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000 - 12000)) \
  --step 2 \
  --headless 1
  echo "running $i"
done
