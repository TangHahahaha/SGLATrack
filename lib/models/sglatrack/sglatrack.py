"""lib.models.sglatrack.sglatrack

该文件定义 **SGLATrack** 的整体网络结构（backbone + head），并提供 `build_sglatrack()` 工厂函数。

整体流程（对应论文 Fig.3）：
1) Backbone（Transformer, one-stream）:
   - 输入：template 图像 + search 图像
   - 输出：融合后的 token 序列特征（包含 template tokens 和 search tokens）
   - SGLATrack 的关键加速点在 backbone 内部（见 base_backbone.py）：
     在饱和层 l* 后，仅执行一个被 selection module 选中的后续层，其余层跳过。

2) Head（预测头）:
   - 将 search tokens reshape 成 2D 特征图
   - 使用 Corner head 或 Center head 输出 bbox / heatmap 等

本 repo 默认使用 Center head（见 experiments/sglatrack/*.yaml）。
"""

import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sglatrack.vit import vit_base_patch16_224
from lib.models.sglatrack.deit import deit_tiny_distilled_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class sglatrack(nn.Module):
    """SGLATrack 主网络（backbone + box_head）"""

    def __init__(self, transformer: nn.Module, box_head: nn.Module, aux_loss: bool = False,
                 head_type: str = "CORNER"):
        """初始化

        参数
        ----
        transformer : nn.Module
            backbone（ViT/DeiT 等），需要实现 forward(z=template, x=search) 并返回 (x, aux_dict)
        box_head : nn.Module
            预测头（CornerPredictor 或 CenterPredictor）
        aux_loss : bool
            是否启用 auxiliary loss（本实现默认 False）
        head_type : str
            "CORNER" 或 "CENTER"，决定 forward_head 的解析方式
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type

        # Center/Corner head 需要知道 search 特征图大小 feat_sz（= search_size/stride）
        if head_type in ("CORNER", "CENTER"):
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        # 如果启用 aux_loss，会复制多个 head（类似 DETR 的深监督）
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn: bool = False,
    ):
        """训练/推理统一 forward（内部会根据 backbone.training 决定是否返回额外 aux 信息）

        参数
        ----
        template : torch.Tensor
            [B, 3, Ht, Wt]
        search : torch.Tensor
            [B, 3, Hs, Ws]
        ce_template_mask / ce_keep_rate :
            可选：Candidate Elimination 相关
        return_last_attn : bool
            是否返回最后一层 attention（本实现通常不需要）

        返回
        ----
        out : dict
            至少包含：
            - pred_boxes: (B, Nq, 4)
            - score_map / size_map / offset_map （取决于 head_type）
            同时会把 backbone 的 aux_dict 合并进去（比如训练时的 cos_tensor/pro）
        """

        # backbone 输出：
        # - x: token 序列特征，形状一般为 [B, N, C]
        # - aux_dict: 训练时可能包含 cos_tensor/pro 等
        x, aux_dict = self.backbone(
            z=template,
            x=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )

        # head 只使用最后一次输出（如果 backbone 返回 list，则取最后一个）
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)

        # 将 backbone 的辅助信息合并到输出，便于 actor 计算额外 loss
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_test(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn: bool = False,
    ):
        """一些脚本会显式调用 forward_test（和 forward 功能类似）

        注意：backbone.forward_test() 通常会启用“层跳过”，且不返回训练用的 cos_tensor。
        """

        x, aux_dict = self.backbone.forward_test(z=template, x=search)

        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature: torch.Tensor, gt_score_map=None):
        """将 backbone 输出的 token 特征送入预测头

        cat_feature : torch.Tensor
            backbone 输出 token，形状一般为 [B, N, C]
            其中 N = N_z + N_x（template token + search token）

        这里的处理：
        - 取最后 N_x 个 token（对应 search region）
        - reshape 成 (B, C, H, W) 送入 head
        """

        # encoder 输出里，最后 feat_len_s 个 token 对应 search region
        enc_opt = cat_feature[:, -self.feat_len_s:]  # [B, N_x, C]

        # reshape 为卷积头输入格式 (B, C, H, W)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # Corner head：预测左上角 / 右下角热力图，再 soft-argmax 得到 box
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)  # -> (cx, cy, w, h)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map,
            }

        elif self.head_type == "CENTER":
            # Center head：输出中心 heatmap + size_map + offset_map
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)

            # bbox 通常已经是 (cx, cy, w, h) 格式（和 box_head.cal_bbox 一致）
            outputs_coord_new = bbox.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map
            }

        raise NotImplementedError


def build_sglatrack(cfg, training: bool = True):
    """工厂函数：根据 cfg 创建 backbone + head，并加载预训练权重"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    # 训练时：如果 cfg.MODEL.PRETRAIN_FILE 指向的是 ImageNet 预训练权重（而不是 sglatrack 自己的 ckpt）
    # 则从 pretrained_models 目录加载
    if cfg.MODEL.PRETRAIN_FILE and ('sglatrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # ------------------------------------------------------------
    # 1) 构建 backbone
    # ------------------------------------------------------------
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1  # ViT 只有 cls token
    elif cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16':
        backbone = deit_tiny_distilled_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 2  # DeiT-distilled 有 cls + distill 两个 token
    else:
        raise NotImplementedError

    # 将 backbone 的 pos_embed/patch_embed 适配到 tracking 输入尺寸
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # ------------------------------------------------------------
    # 2) 构建 head
    # ------------------------------------------------------------
    box_head = build_box_head(cfg, hidden_dim)

    # ------------------------------------------------------------
    # 3) 组装成完整模型
    # ------------------------------------------------------------
    model = sglatrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # ------------------------------------------------------------
    # 4) 如果 cfg.MODEL.PRETRAIN_FILE 指向的是训练好的 SGLATrack ckpt，则加载
    # ------------------------------------------------------------
    if 'sglatrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
