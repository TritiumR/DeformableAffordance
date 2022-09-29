CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --demo_times 10 \
  --exp_name 0928-04-trynine \
  --suffix trynine-step1 \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-3 \
  --model critic \
  --image_size 160 \
  --batch_normalize \

