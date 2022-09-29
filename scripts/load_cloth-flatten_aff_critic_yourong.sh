CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --step 1 \
  --out_logits 1 \
  --exp_name 0927-05-step-1-new_preprocess-expert \
  --suffix tryeight \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 1 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0925-10-tryseven-preprocess-step-1/critic-ckpt-301000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

