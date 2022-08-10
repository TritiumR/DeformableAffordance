CUDA_VISIBLE_DEVICES=1 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/trytwo \
  --process_num 5 \
  --data_num 500 \
  --data_type 5 \
  --headless 1