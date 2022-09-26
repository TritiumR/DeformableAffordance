#!/bin/bash
for ((i=1;i<=1;i++))
do
  CUDA_VISIBLE_DEVICES=$i python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-trynine-step2 \
  --process_num 1 \
  --data_num 5 \
  --data_type 5 \
  --curr_data $(($i * 1000)) \
  --step 2 \
  --headless 1 \
  --save_video_dir ./
  echo "running $i"
done
