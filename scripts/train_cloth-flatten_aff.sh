CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1026-22-tryeleven-aff-with-2:1-195000-step4 \
  --suffix tryeleven-step4 \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4/critic-ckpt-195000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4 \

