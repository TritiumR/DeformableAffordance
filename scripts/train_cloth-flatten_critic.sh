CUDA_VISIBLE_DEVICES=1 python ./softgym/train.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --suffix tryeleven-step2 \
  --num_demos 8000 \
  --demo_times 10 \
  --num_iters 300000 \
  --out_logits 1 \
  --step 2 \
  --exp_name 1105-26-tryeleven-render-online \
  --max_load 1000 \
  --batch 1 \
  --learning_rate 1e-4 \
  --model critic \
  --image_size 160 \
  --unet 1 \
  --use_mask \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-0.h5 \

