CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_dobot_multi_data \
  --policy.dir=/root/data0/openpi0.5/checkpoints/pi05_dobot_multi_data/cyt_test/49999
