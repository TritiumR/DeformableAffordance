CUDA_VISIBLE_DEVICES=1 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryfive \
  --process_num 2 \
  --data_num 1000 \
  --data_type 10 \
  --curr_data 2000 \
  --headless 1