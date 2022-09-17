CUDA_VISIBLE_DEVICES=1 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --step 2 \
  --out_logits 1 \
  --exp_name 0917-08-tryseven-aff_critic-mix \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7000-0916-04-tryseven-step2-many_model-mix_1:1_2-step-2/critic-ckpt-78750.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

