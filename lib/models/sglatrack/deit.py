"""lib.models.sglatrack.deit

在 SGLATrack 中，我们可以选择不同的 ViT-family backbone。
本文件提供了 **DeiT-Tiny Distilled** backbone 的构建函数。

关键点：
- DeiT distilled 版本相比标准 ViT 多了一个 distillation token，
  因此位置编码 `pos_embed` 的长度是 `num_patches + 2`（cls + distill）。
- `DistilledVisionTransformer` 继承 timm 的 `VisionTransformer`，同时混入 `BaseBackbone`，
  使其具备 SGLA 的 token 组织方式与“饱和层后跳层”能力。
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_

from lib.models.sglatrack.base_backbone import BaseBackbone

__all__ = [
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224',
    'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer, BaseBackbone):
    """timm DeiT backbone + SGLATrack BaseBackbone 混入"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # timm 的 DeiT-distilled 默认有 cls_token + dist_token 两个 token
        # 因此 pos_embed 需要覆盖 (num_patches + 2)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, z, x, **kwargs):
        """跟踪任务 forward

        这里直接复用 BaseBackbone.forward：
        - 负责 template/search patch embedding + pos embedding
        - one-stream token concat
        - SGLA 的动态层选择 / 跳层逻辑
        """
        return BaseBackbone.forward(self, z, x, **kwargs)


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """标准 DeiT-Tiny（未 distilled），主要用于对照/备用"""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-Tiny distilled（SGLATrack 默认使用的 backbone）"""

    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        print('Load pretrained model from: ' + str(pretrained))

    return model
