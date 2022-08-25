CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 4000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0825-06-tryfour-with-another-pick \
  --suffix tryfive \
  --max_load 2000 \
  --batch 3 \
  --model critic \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-6300-tryfour/critic-ckpt-300000.h5 \

