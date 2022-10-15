CUDA_VISIBLE_DEVICES=5 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 9000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1015-04-ten_eleven-2:1-step-2-aff_critic-exp \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2/critic-ckpt-300000.h5 \
  --load_critic_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-9000-1011-16-tryten_eleven-online-2:1-step-2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-8000-1014-03-tryeleven-aff-step2/attention-ckpt-400000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-8000-1014-03-tryeleven-aff-step2 \
  --image_size 160 \
  --exp \
  --save_video_dir './test_video/' \
  --headless 1

