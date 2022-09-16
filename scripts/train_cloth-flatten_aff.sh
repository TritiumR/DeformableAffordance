CUDA_VISIBLE_DEVICES=4 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --num_iters 300000 \
  --out_logits 1 \
  --demo_times 10 \
  --exp_name 0916-09-tryseven-aff-not_on_cloth_zero-test \
  --suffix tryseven \
  --max_load 2000 \
  --batch 3 \
  --model aff \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0907-02-tryseven-not_on_cloth_zero-step-1/critic-ckpt-300000.h5 \

