export HF_LEROBOT_HOME=/home/polaris/zch/workspace/openpi0.5/data
export CUDA_VISIBLE_DEVICES=6,7

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_fold_clothes_short_mem --exp-name=sm_890 --tensorboard-enabled
