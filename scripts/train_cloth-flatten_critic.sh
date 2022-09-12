CUDA_VISIBLE_DEVICES=0 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 6000 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --demo_times 10 \
  --exp_name 0912-01-tryseven-step2-many_model-no_mix \
  --suffix tryseven-step2 \
  --max_load 2000 \
  --batch 3 \
  --model critic \
  --load_next_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5

