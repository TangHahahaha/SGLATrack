"""lib.train.actors.sglatrack

Actor = 训练时的“前向 + loss”封装。

在 OSTrack/pytracking 风格的代码库里：
- `net` 负责 forward 得到 pred_dict（分类热力图、bbox 等）
- Actor 负责把 pred_dict 和 gt_dict 组织起来，计算训练损失，并返回用于日志的 status

SGLATrack 相比普通跟踪器额外多了一个 **Similarity-Guided Layer Adaptation** 的监督项：
- backbone 在训练 forward 中会返回：
  - `cos_tensor`: 每个候选后续层与饱和层特征的余弦相似度（用于生成“伪标签”y）
  - `pro`: 选择模块 MLP 输出的概率 ŷ
- Actor 使用 `cos_tensor` 的 argmax 生成 one-hot 目标 `pro_target`，
  再用 L1 距离约束 `pro ≈ pro_target`，对应论文 Eq.(6)-(7) 的 layer-wise similarity loss
"""

from . import BaseActor

import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class sglatrackActor(BaseActor):
    """SGLATrack 的训练 Actor"""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """一次迭代：forward + loss

        输入 data 来自 dataloader（见 lib/train/data/...），典型字段包括：
        - template_images: (N_t, B, 3, Ht, Wt)
        - search_images  : (N_s, B, 3, Hs, Ws)
        - template_anno / search_anno: GT bbox 等

        返回：
        - loss: 标量，用于反向传播
        - status: dict，用于 logger/wandb 记录
        """
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        """组装输入并调用网络 forward"""

        # 当前实现仅支持 1 template + 1 search（Siamese/one-stream 跟踪常规设置）
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        # --------------------------
        # 1) 取出模板图像
        # --------------------------
        template_list = []
        for i in range(self.settings.num_template):
            # data['template_images'][i] 的形状为 (1, B, 3, H, W) 或类似，
            # 这里 view(-1, ...) 把 (N_t, B, ...) 展平成 (B, ...)
            template_img_i = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:]
            )  # (B, 3, 128, 128) in default configs
            template_list.append(template_img_i)

        # --------------------------
        # 2) 取出搜索图像
        # --------------------------
        search_img = data['search_images'][0].view(
            -1, *data['search_images'].shape[2:]
        )  # (B, 3, Hs, Ws)，默认 (B, 3, 256, 256)

        # --------------------------
        # 3) Candidate Elimination (CE) 相关（可选）
        #    - 若启用，会根据 template bbox 生成 mask，引导 backbone 更关注目标区域
        # --------------------------
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_list[0].shape[0],
                template_list[0].device,
                data['template_anno'][0]
            )

            # CE 的 keep_rate 通常有 warmup 策略，随 epoch 逐渐调整
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        # 兼容：只有 1 个 template 时直接取 tensor
        if len(template_list) == 1:
            template_list = template_list[0]

        # --------------------------
        # 4) 网络 forward
        # --------------------------
        out_dict = self.net(
            template=template_list,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=False
        )
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """计算总损失

        pred_dict 关键字段：
        - pred_boxes : (B, Nq, 4) 预测框 (cx, cy, w, h) 或类似格式
        - score_map  : 分类/中心热力图
        - size_map / offset_map : Center head 的额外输出
        - cos_tensor : (B, K) 每个候选后续层与饱和层输出的余弦相似度（训练时有）
        - pro        : (B, K) 选择模块输出的概率（训练时有）

        gt_dict 关键字段：
        - search_anno : GT bbox，通常是 (Ns, B, 4) 的序列，取最后一帧 [-1]
        """

        # ------------------------------------------------------------------
        # 1) 生成 GT 热力图（center-based head 的监督信号）
        # ------------------------------------------------------------------
        gt_bbox = gt_dict['search_anno'][-1]  # (B, 4) 这里一般是 (x, y, w, h) 归一化坐标
        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B, 1, H, W)

        # ------------------------------------------------------------------
        # 2) bbox regression 相关 loss：GIoU + L1
        # ------------------------------------------------------------------
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B*Nq, 4) -> (x1,y1,x2,y2)

        # GT bbox 扩展到与 queries 数量一致
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)) \
            .view(-1, 4).clamp(min=0.0, max=1.0)

        # giou / iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            # 某些极端情况下可能会报错（比如框退化），这里做容错避免训练中断
            giou_loss = torch.tensor(0.0).cuda()
            iou = torch.tensor(0.0).cuda()

        # L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        # ------------------------------------------------------------------
        # 3) 分类/定位热力图 loss（Focal loss）
        # ------------------------------------------------------------------
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # ------------------------------------------------------------------
        # 4) Similarity-guided loss（论文 Eq.(6)-(7)）
        #
        # backbone 在训练 forward 中会返回：
        # - cos_tensor: 每个候选层与饱和层输出的相似度
        # - pro: 选择模块对每个候选层的选择概率
        #
        # 我们用 cos_tensor.argmax 得到“最相似的候选层”作为伪标签 y（one-hot），
        # 再用 L1(pro, y) 监督选择模块。
        # ------------------------------------------------------------------
        cos_tensor = pred_dict['cos_tensor']          # [B, K]
        indices = torch.argmax(cos_tensor, dim=1)     # [B]，每个样本最相似的候选层索引

        pro_target = torch.zeros_like(cos_tensor)     # [B, K]
        pro_target.scatter_(1, indices.unsqueeze(1), 1)

        pro = pred_dict['pro']                        # [B, K]，Sigmoid 后的概率
        pro_loss = self.objective['l1'](pro, pro_target)

        # ------------------------------------------------------------------
        # 5) 总损失加权求和（论文 Eq.(8)）
        #    这里的 0.2 就是论文中的 γ（相似度损失权重）
        # ------------------------------------------------------------------
        loss = (
            self.loss_weight['giou'] * giou_loss
            + self.loss_weight['l1'] * l1_loss
            + self.loss_weight['focal'] * location_loss
            + 0.2 * pro_loss
        )

        if return_status:
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "pro_loss": pro_loss.item(),
                "IoU": mean_iou.item()
            }
            return loss, status

        return loss
