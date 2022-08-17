CUDA_VISIBLE_DEVICES=1 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryfour \
  --process_num 3 \
  --data_num 1000 \
  --data_type 10 \
  --curr_data 300 \
  --headless 1