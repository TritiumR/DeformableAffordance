CUDA_VISIBLE_DEVICES=3 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --num_demos 7900 \
  --num_iters 400000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 1105-03-tryseven-no_perturb-on_unet-aff-step1 \
  --suffix tryseven-step1-S \
  --max_load 2000 \
  --batch 20 \
  --model aff \
  --image_size 160 \
  --no_perturb \
  --unet 0 \
  --load_critic_dir checkpoints/rope-configuration-Aff_Critic-7900-1103-05-no_unet-tryseven-S-step1-no_perturb-step-1/critic-ckpt-300000.h5 \

