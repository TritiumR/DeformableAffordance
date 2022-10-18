CUDA_VISIBLE_DEVICES=3 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 10000 \
  --data_type 2 \
  --critic_type 5 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1018-08-online-step-2-both-0.8 \
  --mode both \
  --process_num 1 \
  --critic_depth 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1/critic-ckpt-301000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-10000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1 \
  --image_size 160 \
  --headless 1

