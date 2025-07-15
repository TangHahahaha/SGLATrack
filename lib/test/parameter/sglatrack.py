"""lib.test.parameter.sglatrack

推理/评测时的参数入口。

评测框架会通过 `parameters(yaml_name)` 获取：
- cfg（由 experiments/sglatrack/{yaml_name}.yaml 覆盖默认 cfg）
- template/search 的裁剪参数
- checkpoint 路径（根据 save_dir + epoch 拼接）

使用示例（见 README）：
    python tracking/test.py --tracker_param sglatrack --dataset uav123 ...

其中 tracker_param=sglatrack 会最终调用到这里。
"""

import os

from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.config.sglatrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    """根据 yaml 名称构造 TrackerParams"""

    params = TrackerParams()

    # env_settings() 会读取 lib/test/evaluation/local.py 中的路径配置
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir

    # 1) 读取实验配置：experiments/sglatrack/{yaml_name}.yaml
    yaml_file = os.path.join(prj_dir, f'experiments/sglatrack/{yaml_name}.yaml')
    update_config_from_file(yaml_file)

    params.cfg = cfg
    print("test config: ", cfg)

    # 2) template/search 裁剪参数
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # 3) checkpoint 路径：
    # 默认训练脚本会把权重存到：
    #   {save_dir}/checkpoints/train/sglatrack/{yaml_name}/sglatrack_epXXXX.pth.tar
    params.checkpoint = os.path.join(
        save_dir,
        "checkpoints/train/sglatrack/%s/sglatrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH)
    )

    # 4) 可选：是否保存所有 query 的预测框
    params.save_all_boxes = False

    return params
