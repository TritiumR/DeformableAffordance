CUDA_VISIBLE_DEVICES=1 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-trysix \
  --process_num 3 \
  --data_num 3000 \
  --data_type 10 \
  --curr_data 0 \
  --headless 1