CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1018-06-tryeleven-aff-step3 \
  --suffix tryeleven-step2 \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1016-04-tryeleven-online_step3-step-3/critic-ckpt-210000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1016-04-tryeleven-online_step3-step-3 \

