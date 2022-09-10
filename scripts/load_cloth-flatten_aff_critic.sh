CUDA_VISIBLE_DEVICES=0 python ./softgym/test.py \
  --env_name ClothFlatten \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 7500 \
  --out_logits 1 \
  --exp_name 0910-01-tryseven-aff_critic \
  --suffix tryseven \
  --process_num 1 \
  --num_test 20 \
  --load_critic_dir checkpoints/cloth-flatten-Aff_Critic-7500-0907-02-tryseven-not_on_cloth_zero-step-1/critic-ckpt-300000.h5 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5 \
  --save_video_dir './test_video/' \
  --headless 1

