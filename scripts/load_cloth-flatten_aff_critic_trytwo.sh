CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 2500 \
  --out_logits 1 \
  --exp_name 0815-01-trytwo-critic \
  --suffix trytwo \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-2500/critic-ckpt-50000.h5 \
  --critic_pick \
  --headless 1

