CUDA_VISIBLE_DEVICES=0 python softgym/collect.py \
  --env_name ClothFlatten \
  --path ./data/cloth-flatten-tryfour \
  --process_num 3 \
  --data_num 100 \
  --data_type 10 \
  --headless 1