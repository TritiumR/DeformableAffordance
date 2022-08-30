CUDA_VISIBLE_DEVICES=0 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryseven \
  --process_num 2 \
  --data_num 3000 \
  --data_type 5 \
  --curr_data 0 \
  --headless 1