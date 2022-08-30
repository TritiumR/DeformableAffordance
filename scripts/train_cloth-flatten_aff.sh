CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6300 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0825-05-tryfour-aff \
  --suffix tryfour \
  --max_load 2000 \
  --batch 3 \
  --model aff \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-4000-tryfive-step-1/critic-ckpt-200000.h5 \

