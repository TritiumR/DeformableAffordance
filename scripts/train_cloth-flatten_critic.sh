CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 1 \
  --demo_times 10 \
  --exp_name 0917-02-tryseven-step1-no_perturb \
  --suffix tryseven \
  --max_load 3000 \
  --batch 4 \
  --learning_rate 2e-4 \
  --model critic \
  --no_perturb

