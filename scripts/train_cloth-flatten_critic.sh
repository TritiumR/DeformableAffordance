CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 1236 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0903-02-tryseven-not_on_cloth_zero \
  --suffix tryseven \
  --max_load 1236 \
  --batch 3 \
  --model critic \

