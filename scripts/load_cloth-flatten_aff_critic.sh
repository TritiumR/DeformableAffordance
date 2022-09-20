CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 0919-4-step-2-no_mix-expert-1.5-1.5-1.5 \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7000-0917-01-tryseven-step2-many_model-no_mix-step-2/critic-ckpt-165000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

