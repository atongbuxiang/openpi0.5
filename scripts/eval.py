import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

def main():
    # 获取配置
    config = _config.get_config("pi0_fast_droid")
    
    # 下载或加载模型检查点
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

    # 创建训练好的策略模型
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 使用 DROID 格式的示例输入进行推理
    example = droid_policy.make_droid_example()
    result = policy.infer(example)

    # 删除策略以释放内存
    del policy

    # 打印输出结果
    print("Actions shape:", result["actions"].shape)
    print("Actions:", result["actions"])

if __name__ == "__main__":
    main()
