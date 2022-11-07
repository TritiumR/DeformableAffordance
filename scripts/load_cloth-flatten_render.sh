CUDA_VISIBLE_DEVICES=6 python ./softgym/test_render.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --step 1 \
  --test_step 1 \
  --out_logits 1 \
  --exp_name 1107-21-by-step3-both-online \
  --process_num 1 \
  --num_test 100 \
  --critic_depth 1 \
  --pick_num 4 \
  --place_num 5 \
  --without_global \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/critic-online-ckpt-9500.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-10000-1019-08-online-step-3-both-0.7-step-1/attention-online-ckpt-9500.h5 \
  --image_size 160 \
  --headless 1 \
  --use_mask

