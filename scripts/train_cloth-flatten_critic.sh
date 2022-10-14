CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step2 \
  --num_demos 9000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1013-04-tryten_eleven-online_render \
  --max_load 2000 \
  --batch 1 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1 \
  --model critic \
  --image_size 160 \

