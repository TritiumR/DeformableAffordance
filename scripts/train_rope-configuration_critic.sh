CUDA_VISIBLE_DEVICES=2 python ./softgym/train.py \
  --env_name RopeConfiguration \
  --task rope-configuration \
  --agent aff_critic \
  --suffix tryseven-step3-S \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 200000 \
  --out_logits 1 \
  --step 3 \
  --exp_name 1108-01-tryseven-S-no_unet-step3-no_perturb \
  --max_load 5000 \
  --batch 20 \
  --unet 0 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --load_aff_dir checkpoints/rope-configuration-Aff_Critic-3000-1107-23-tryseven-no_unet-aff-step2-step-1/attention-ckpt-300000.h5 \
  --no_perturb \
