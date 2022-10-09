#!/bin/bash
for ((i=4;i<=5;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryeleven-step2 \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 4000 - 16000)) \
  --step 2 \
  --headless 1 &
  echo "running $i"
done
