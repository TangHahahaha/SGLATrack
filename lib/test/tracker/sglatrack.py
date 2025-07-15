"""lib.test.tracker.sglatrack

该文件实现 **推理/评测阶段** 的 tracker 封装（继承 BaseTracker）。

运行逻辑（经典 Siamese/one-stream tracker 流程）：
1) initialize():
   - 从首帧裁剪 template patch（通常以 init_bbox 为中心）
   - 做预处理（归一化/转 tensor）并缓存 template feature（这里直接缓存 template tensor）
2) track():
   - 每帧从当前 state 裁剪 search patch
   - 调用 network(template, search) 得到 heatmap + bbox 等
   - 对 heatmap 乘 Hann window 做平滑（抑制边缘响应）
   - 由 Center head 解码 bbox
   - 把 search 坐标系的预测框映射回原图坐标系，更新 state

注意：
- SGLATrack 的加速主要在 backbone 内部：forward_test() 会在饱和层后只执行一个选中的后续层。
- 推理时 batch 通常为 1（逐视频逐帧），因此 BaseBackbone.forward_test 里用了 break 优化。
"""

import os

import cv2
import torch

from lib.models.sglatrack import build_sglatrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class sglatrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(sglatrack, self).__init__(params)

        # 1) 构建网络（training=False 会走 backbone 的推理路径）
        network = build_sglatrack(params.cfg, training=False)

        # 2) 加载 checkpoint
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        self.cfg = params.cfg
        self.network = network.cuda().eval()

        # 预处理：将 numpy 图像 + mask 转成 torch tensor
        self.preprocessor = Preprocessor()

        self.state = None  # 当前帧的 bbox (x, y, w, h)，原图坐标系

        # search 特征图大小（用于生成 Hann window）
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # Hann window：对 response map 做平滑，抑制边缘
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # debug 可视化
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                os.makedirs(self.save_dir, exist_ok=True)
            else:
                self._init_visdom(None, 1)

        # 是否保存所有 query 的预测框（有些论文会分析多 query 输出）
        self.save_all_boxes = params.save_all_boxes

        # 缓存 template（这里直接缓存 template tensor，不做动态更新）
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        """初始化：处理首帧 template"""

        # 从首帧裁剪 template patch
        # 返回：
        # - z_patch_arr: 裁剪后的 template 图像 (H,W,3)
        # - resize_factor: 从原图到 template patch 的缩放倍率
        # - z_amask_arr: 对应的 mask
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size
        )
        self.z_patch_arr = z_patch_arr

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        # 缓存 template tensor（网络 forward 需要 tensor）
        with torch.no_grad():
            self.z_dict1 = template

        # Candidate Elimination (CE) 可选：根据 init_bbox 生成 template mask
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info['init_bbox'],
                resize_factor,
                template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # 记录状态
        self.state = info['init_bbox']
        self.frame_id = 0

        if self.save_all_boxes:
            # 第 0 帧直接把 init_bbox 复制成多 query 形式保存
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        """逐帧跟踪"""

        H, W, _ = image.shape
        self.frame_id += 1

        # 1) 裁剪 search patch（以当前 state 为中心）
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.params.search_factor,
            output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # 2) 网络前向
        with torch.no_grad():
            out_dict = self.network(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z
            )

        # 3) Center head 输出的 score_map 做 Hann window 平滑
        pred_score_map = out_dict['score_map']        # (B,1,Hf,Wf) or similar
        response = self.output_window * pred_score_map

        # 4) 解码 bbox（cx,cy,w,h），单位是 feature map / 归一化尺度
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)

        # 5) 多 query 情况下，简单取均值作为最终预测（baseline 策略）
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()

        # 6) 将 search patch 坐标系下的 box 映射回原图坐标系，并做边界裁剪
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # 7) debug 可视化
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                              color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz),
                                     'heatmap', 1, 'score_map_hann')

                # 如果启用了 token pruning 等，会在 out_dict 中存 removed_indexes_s（可视化被 mask 的区域）
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = [ri.cpu().numpy() for ri in out_dict['removed_indexes_s']]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1),
                                         'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            # 保存所有 query 的预测框（映射回原图坐标系）
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        """将 search patch 坐标系下的 box 映射回原图坐标系"""

        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]

        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        """batch 版本 map_box_back（pred_box: (N,4)）"""

        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]

        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        """调试用：注册 attention hook，保存 encoder attention 权重"""
        enc_attn_weights = []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return sglatrack
