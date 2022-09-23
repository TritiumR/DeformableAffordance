CUDA_VISIBLE_DEVICES=0 python softgym/check_data.py \
  --path ./data/cloth-flatten-tryeight-step2 \
  --iepisode 10 \
  --step 2 \
  --render_demo \
  --need_aff \
  --load_aff_dir checkpoints/cloth-flatten-Aff_Critic-7500-0909-02-tryseven-aff-not_on_cloth_zero-step-1/attention-ckpt-300000.h5