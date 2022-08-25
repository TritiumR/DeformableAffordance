CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 3300 \
  --out_logits 1 \
  --exp_name 0825-04-tryfour-for-critic \
  --suffix tryfive \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-2000-tryfive/critic-ckpt-100000.h5 \
  --expert_pick \
  --headless 1

