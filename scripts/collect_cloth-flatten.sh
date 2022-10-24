#!/bin/bash
for ((i=4;i<=4;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryeleven-step3 \
  --process_num 2 \
  --data_num 2000 \
  --data_type 10 \
  --curr_data $(($i * 2000)) \
  --step 3 \
  --headless 1 &
  echo "running $i"
done
