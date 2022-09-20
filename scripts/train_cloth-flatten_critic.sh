CUDA_VISIBLE_DEVICES=7 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --demo_times 10 \
  --exp_name 0920-12-tryeight-step2-no-mix-depth2 \
  --suffix tryeight-step2 \
  --max_load 3000 \
  --batch 4 \
  --critic_depth 2 \
  --learning_rate 2e-4 \
  --model critic \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5 \

