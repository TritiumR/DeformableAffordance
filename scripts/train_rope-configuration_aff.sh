CUDA_VISIBLE_DEVICES=6 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 3000 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1107-23-tryseven-no_unet-aff-step2 \
  --suffix tryseven-step2-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-8000-1106-07-no_unet-tryseven-S-step2-no_perturb-step-2/critic-ckpt-270000.h5 \

