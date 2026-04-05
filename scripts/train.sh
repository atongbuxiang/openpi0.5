export HF_LEROBOT_HOME=/data0/lerobot_dataset
export CUDA_VISIBLE_DEVICES=6

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_dobot_multi_data \
  --exp-name=cyt_test 
