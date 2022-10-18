CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step3 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1018-05-tryeleven-online-17000-step3-2:1 \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1/attention-online-ckpt-17000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-20000-1010-08-test_online-step-3-0.7-step-1 \
  --model critic \
  --image_size 160 \

