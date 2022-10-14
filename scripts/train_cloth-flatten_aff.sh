CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1014-03-tryeleven-aff-step2 \
  --suffix tryeleven-step2 \
  --max_load 2000 \
  --batch 24 \
  --model aff \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2/critic-ckpt-255000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2 \

