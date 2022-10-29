CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step4 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 200000 \
  --out_logits 1 \
  --step 4 \
  --exp_name 1029-08-tryeleven-2:1-mix1 \
  --max_load 3000 \
  --batch 24 \
  --critic_depth 1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-9500.h5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4/critic-ckpt-195000.h5 \
  --learning_rate 5e-5 \
  --model critic \
  --image_size 160 \

