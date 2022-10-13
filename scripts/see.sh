CUDA_VISIBLE_DEVICES=-1 python ./softgym/see.py \
  --task cloth-flatten \
  --agent aff_critic \
  --num_demos 5 \
  --demo_times 10 \
  --step 2 \
  --out_logits 1 \
  --exp_name 1013-03-step-2-online-see\
  --suffix tryeleven-step2 \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1/attention-online-ckpt-7000.h5 \
  --load_aff_mean_std_dir checkpoints/cloth-flatten-Aff_Critic-50000-1010-08-test_online-step-2-0.8-step-1 \
  --image_size 160 \

