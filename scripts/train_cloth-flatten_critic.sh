CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --demo_times 10 \
  --exp_name 0926-03-tryeight-preprocess \
  --suffix tryeight-step2 \
  --max_load 1000 \
  --batch 4 \
  --critic_depth 3 \
  --learning_rate 3e-4 \
  --model critic \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5 \

