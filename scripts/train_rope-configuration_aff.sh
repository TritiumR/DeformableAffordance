CUDA_VISIBLE_DEVICES=6 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 8000 \
  --num_iters 200000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1111-06-tryseven-no-unet-aff-step4-with-140000 \
  --suffix tryseven-step4-S \
  --max_load 8000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1110-15-tryseven-S-no-unet-step4-step-4/critic-ckpt-140000.h5 \

