"""lib.models.sglatrack.utils

一些与 one-stream 跟踪相关的 token 工具函数。

在 ViT 跟踪器中，我们通常把：
- template token 序列 Z  （来自模板 patch embedding）
- search token 序列 X     （来自搜索区域 patch embedding）

拼成一个长序列送入 Transformer。不同的拼接方式会影响 token 的相对位置与注意力结构，
因此这里提供了多种 combine/recover 策略：

- direct：最常见，直接 [Z, X]
- template_central：把 template 插入到 search token 的中间
- partition：对 template token 做一次简单的“窗口重排/拼接”（用于某些变体/对照实验）
"""

import math

import torch
import torch.nn.functional as F


def combine_tokens(template_tokens: torch.Tensor,
                   search_tokens: torch.Tensor,
                   mode: str = 'direct',
                   return_res: bool = False):
    """将 template/search token 合并成 one-stream 序列

    参数
    ----
    template_tokens : torch.Tensor
        [B, N_z, C]
    search_tokens : torch.Tensor
        [B, N_x, C]
    mode : str
        'direct' | 'template_central' | 'partition'
    return_res : bool
        仅在 partition 模式下使用：返回新的 (H, W) 估计

    返回
    ----
    merged_feature : torch.Tensor
        [B, N_z + N_x, C]（或 partition 模式下 token 顺序会变化）
    """

    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        # [Z, X]
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)

    elif mode == 'template_central':
        # 把 template 插入 search token 中间：
        # [X_first, Z, X_second]
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)

    elif mode == 'partition':
        # 这个模式会把 template token reshape 成 2D 网格，做简单“重排/拼接”
        feat_size_s = int(math.sqrt(len_s))  # search token 的边长
        feat_size_t = int(math.sqrt(len_t))  # template token 的边长
        window_size = math.ceil(feat_size_t / 2.)

        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)

        # pad 到 window_size 的整数倍（这里只 pad 了 top）
        pad_l = pad_b = pad_r = 0
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # 重新组织 token：把 template 的两段拼到一起（实现细节来自原库）
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)

        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)

        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # 对某些 backbone（如 Swin）可能需要 merged 的 2D 尺寸
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError(f"Unknown cat mode: {mode}")

    return merged_feature


def recover_tokens(merged_tokens: torch.Tensor,
                   len_template_token: int,
                   len_search_token: int,
                   mode: str = 'direct'):
    """将合并后的 token 恢复为 [Z, X] 的顺序（某些 mode 会打乱顺序）"""

    if mode == 'direct':
        recovered_tokens = merged_tokens

    elif mode == 'template_central':
        # combine 时是 [X_first, Z, X_second]，这里恢复成 [Z, X_first, X_second]
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)

    elif mode == 'partition':
        # partition 模式下当前实现不做恢复（直接返回）
        recovered_tokens = merged_tokens

    else:
        raise NotImplementedError(f"Unknown cat mode: {mode}")

    return recovered_tokens


def window_partition(x: torch.Tensor, window_size: int):
    """把 (B,H,W,C) 切成多个窗口 (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """window_partition 的逆操作：把窗口拼回 (B,H,W,C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
