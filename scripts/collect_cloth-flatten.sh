CUDA_VISIBLE_DEVICES=0 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryseven \
  --process_num 3 \
  --data_num 2500 \
  --data_type 10 \
  --curr_data 0 \
  --step 1 \
  --headless 1