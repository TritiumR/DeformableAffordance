CUDA_VISIBLE_DEVICES=0 python ./softgym/test_render.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1105-31-render-step2 \
  --process_num 1 \
  --num_test 100 \
  --critic_depth 1 \
  --pick_num 1 \
  --place_num 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1/critic-online-ckpt-1000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-20000-1017-06-online-step-3-both-0.7-step-1/attention-online-ckpt-7000.h5 \
  --image_size 160 \
  --headless 1 \
  --use_mask

