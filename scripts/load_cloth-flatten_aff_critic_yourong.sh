CUDA_VISIBLE_DEVICES=1 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --step 1 \
  --out_logits 1 \
  --exp_name 0918-2-step-1-no_perturb-1.5-1.5-1.0 \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0917-02-tryseven-step1-no_perturb-step-1/critic-ckpt-300000.h5 \
  --expert_pick \
  --save_video_dir './test_video/' \
  --headless 1

