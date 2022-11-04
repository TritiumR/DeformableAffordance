CUDA_VISIBLE_DEVICES=0 python ./softgym/test_render.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 1 \
  --out_logits 1 \
  --exp_name 1104-11-render-sep \
  --process_num 1 \
  --num_test 100 \
  --critic_depth 1 \
  --pick_num 4 \
  --place_num 5 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-1001-04-tryten-step-1/critic-ckpt-301000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-7000.h5 \
  --image_size 160 \
  --headless 1 \
  --use_mask

