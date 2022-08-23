CUDA_VISIBLE_DEVICES=1 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 3300 \
  --out_logits 1 \
  --exp_name 0822-01-tryfour-expert-for_aff \
  --suffix tryfour \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-3300-tryfour/critic-ckpt-300000.h5 \
  --expert_pick \
  --headless 1

