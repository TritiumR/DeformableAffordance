CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0930-06-trynine-aff \
  --suffix trynine-step1 \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-6000-0928-01-trynine-step-1/critic-ckpt-106000.h5 \

