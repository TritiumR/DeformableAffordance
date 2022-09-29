CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 8000 \
  --step 2 \
  --out_logits 1 \
  --exp_name 0927-02-step-2-depth-3-expert \
  --suffix tryeight \
  --process_num 1 \
  --num_test 20 \
  --critic_depth 3 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-8000-0922-1-tryeight-step2-no-mix-depth3-step-2/critic-ckpt-240000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

