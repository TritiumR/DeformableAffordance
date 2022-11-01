CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1101-08-without_global \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --without_global \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1030-17-tryten-without_global-step-1/critic-ckpt-210000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-9000.h5 \
  --image_size 160 \
  --headless 1 \
  --use_mask

