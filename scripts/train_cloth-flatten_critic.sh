CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step4 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 400000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1021-03-tryeleven-online-5000-step4-1:1 \
  --max_load 2000 \
  --batch 20 \
  --critic_depth 1 \
  --learning_rate 1e-4 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-5000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1 \
  --model critic \
  --image_size 160 \

