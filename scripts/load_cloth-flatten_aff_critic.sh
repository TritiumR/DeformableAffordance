CUDA_VISIBLE_DEVICES=4 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 0916-07-tryseven-aff_critic-no_mix \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7000-0913-01-tryseven-step2-many_model-no_mix-step-2/critic-ckpt-225000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

