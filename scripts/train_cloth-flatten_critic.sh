CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step3 \
  --num_demos 12000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1026-01-tryeleven-2:1-mix2 \
  --max_load 3000 \
  --batch 1 \
  --critic_depth 1 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \

