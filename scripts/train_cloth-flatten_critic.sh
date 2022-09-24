CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --step 2 \
  --demo_times 10 \
  --exp_name 0924-03-tryeight-no_mix-batch_normalize \
  --suffix tryeight-step2 \
  --max_load 1000 \
  --batch 5 \
  --critic_depth 1 \
  --learning_rate 5e-4 \
  --model critic \
  --batch_normalize \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5 \

