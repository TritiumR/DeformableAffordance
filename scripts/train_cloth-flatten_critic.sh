CUDA_VISIBLE_DEVICES=5 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step5 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 200000 \
  --out_logits 1 \
  --step 5 \
  --exp_name 1102-01-tryeleven-step5-continue \
  --max_load 3000 \
  --batch 20 \
  --learning_rate 5e-5 \
  --model critic \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-07-tryeleven-2:1-mix1-step-5/critic-ckpt-200000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1028-03-online-step-5-aff-0.6-step-1/attention-online-ckpt-9500.h5

