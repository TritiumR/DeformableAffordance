CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 4 \
  --out_logits 1 \
  --exp_name 1029-06-online-5500-aff-195000-critic-step-4 \
  --process_num 1 \
  --num_test 30 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1024-02-tryeleven-with-online-aff-9500-2:1-mix1-step4-step-4/critic-ckpt-195000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1028-03-online-step-5-aff-0.6-step-1/attention-online-ckpt-5500.h5 \
  --image_size 160 \
  --headless 1

