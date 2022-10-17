CUDA_VISIBLE_DEVICES=7 python ./softgym/train_online.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_online 20000 \
  --data_type 4 \
  --step 3 \
  --out_logits 1 \
  --exp_name 1017-06-online-step-3-both-0.7 \
  --mode both \
  --process_num 1 \
  --critic_depth 1 \
  --learning_rate 5e-5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1014-03-tryeleven-aff-step2/attention-ckpt-400000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1014-03-tryeleven-aff-step2 \
  --image_size 160 \
  --headless 1

