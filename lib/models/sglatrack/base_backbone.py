"""lib.models.sglatrack.base_backbone

本文件实现 **SGLATrack** 论文中的 *Similarity-Guided Layer Adaptation (SGLA)* 核心逻辑。

在 SGLATrack 中，我们观察到轻量 ViT 跟踪器在较深层会出现“表示饱和/冗余”：  
- 浅层：特征变化大（更像在做外观/细节对齐）  
- 深层：相邻层特征变化小（很多层学到的表示高度相似）

因此论文提出：当特征在某个“饱和层 l*”后，只保留 **一个** 后续层作为代表层，
其它后续层全部跳过，以获得更好的 **精度-速度** 折中。

代码实现要点（与论文 3.2 对齐）：
1) `start_layer`：0-based 的“饱和层前一层索引”
   - 代码中 `start_layer = 5` 表示：在第 6 层(1-based)处视为饱和层 l*  
2) `ThreeLayerMLP`：选择模块 M（论文中的 selection module）
   - 输入：饱和层输出 `X^{l*}` 的 **第 0 维通道**在 token 维度上的向量（形状 [B, N]）
     - 这对应论文中的 `e_1^T X^{l*}`（只取某一维通道作为轻量化输入）
   - 输出：每个“候选后续层”的被选择概率 `ŷ`（形状 [B, K]）
3) 训练时（`forward_`）：
   - 先正常跑到饱和层得到 `mid = X^{l*}`
   - 用 MLP 产生概率 `pro`
   - 对每个样本选择概率最大的后续层（默认只保留 1 层）
   - 额外计算 `cos_tensor`：用于构造监督信号（哪个后续层与 `mid` 最相似）
4) 测试/跟踪时（`forward_test`）：
   - batch 通常为 1（逐帧跟踪），因此选择出后续层后直接执行并 `break` 退出循环

注意：
- 本实现默认 ViT depth=12，因此在若干地方写死了 `12`。若你更换了 backbone 深度，
  需要同步修改 `output_dim` / `cos_tensor` 的维度计算逻辑。
- `forward_test()` 中的 `break` 依赖 “B=1”。如果你想用 batch>1 做推理，需要移除 break，
  并像训练那样对不同样本分别执行各自选中的层。
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.sglatrack.utils import combine_tokens, recover_tokens

# -----------------------------------------------------------------------------
# 这些超参数对应论文中的“饱和层 l*”与“保留层数量”
# -----------------------------------------------------------------------------

# 论文设置：在饱和层之后，仅保留 1 个最优层（enabled_layer_num=1）
enabled_layer_num = 1

# start_layer 使用 0-based 索引：
# - i < start_layer：全部执行（对应 1..start_layer 层）
# - i == start_layer：视为饱和层 l*（论文中默认 l*=6，因此 start_layer=5）
# - i > start_layer：这些是候选后续层，最终只执行其中一个（或 top-k 个）
start_layer = 5  # true saturated layer index (1-based) = start_layer + 1


class ThreeLayerMLP(nn.Module):
    """选择模块（Selection Module, 论文中的 M）

    论文中 M 是一个简单的 MLP，用来根据饱和层输出特征 `X^{l*}` 预测要保留的后续层。

    本实现的输入设计：
    - 假设 token 总数 N = N_z + N_x（模板 token + 搜索 token）
    - 取 `X^{l*}` 的 **第 0 个通道**在 token 维度上的向量：`x[:, :, 0]`，形状为 [B, N]
      这样输入是 1D 向量，计算量极小（与论文描述一致）。

    参数
    ----
    input_dim : int
        输入维度 N（token 数）。
        例如：template=128, search=256, patch=16 => N_z=64, N_x=256, N=320
    output_dim : int
        候选后续层数量 K = L - l*（L=总层数，l*=饱和层索引 1-based）
    hidden_dim : int
        隐藏层维度（论文中默认 160）
    """

    def __init__(self, input_dim=320, output_dim=6, hidden_dim=160):
        super().__init__()

        # 注意：这里的 “ThreeLayer” 更像是 “两层全连接 + 激活/输出层” 的组合说法：
        # input -> fc1 -> ReLU -> fc2 -> Sigmoid
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：输出每个候选后续层的选择概率

        输入
        ----
        x : torch.Tensor
            [B, N]，来自饱和层输出的 1D token 表示（如 `X^{l*}[:, :, 0]`）

        返回
        ----
        pro : torch.Tensor
            [B, K]，每个候选后续层的选择概率（Sigmoid 后的值）
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        pro = self.sigmoid(x)
        return pro


class BaseBackbone(nn.Module):
    """SGLATrack 的 backbone 基类

    该类提供：
    - template/search 的 patch embedding + position embedding 处理
    - token 拼接/恢复（one-stream 跟踪范式）
    - 核心：在饱和层后执行“动态层选择”，跳过冗余 transformer blocks
    """

    def __init__(self):
        super().__init__()

        # 原始 ViT 的位置编码参数（会在 finetune_track 里被拆成 z/x 两份）
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        # token 拼接方式，见 combine_tokens()（direct / template_central / partition）
        self.cat_mode = 'direct'

        # 分别给模板/搜索区域用的 pos embed（resize 后得到）
        self.pos_embed_z = None
        self.pos_embed_x = None

        # 如果启用 segment embedding，用于区分 template token / search token
        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        # 是否返回中间层特征（默认 False）
        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):
        """将 ImageNet 预训练 ViT “适配”为跟踪任务的输入尺寸与 token 组织方式

        主要做三件事：
        1) 根据 cfg 中的 stride/输入尺寸，重新构建 patch_embed（可能需要插值权重）
        2) 将原始 `pos_embed` 插值成 template/search 各自的 `pos_embed_z/pos_embed_x`
        3) 初始化选择模块（self.MLP），用于饱和层后的动态层选择
        """

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)       # e.g. (256, 256)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)   # e.g. (128, 128)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE          # e.g. 16

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # ------------------------------------------------------------------
        # 1) 重新构建 patch embedding
        #    - timm 的 ViT 默认 patch_size=16
        #    - 若跟踪输入使用不同 stride，这里会对 conv kernel 做插值以复用预训练权重
        # ------------------------------------------------------------------
        if True:
            print('Timm patch embedding is reload!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    # 将预训练的 patch_embed 卷积核插值到 new_patch_size
                    param = nn.functional.interpolate(
                        param,
                        size=(new_patch_size, new_patch_size),
                        mode='bicubic',
                        align_corners=False
                    )
                    param = nn.Parameter(param)
                old_patch_embed[name] = param

            # 新建 PatchEmbed，并把插值后的权重/偏置赋回去
            self.patch_embed = PatchEmbed(
                img_size=self.img_size,
                patch_size=new_patch_size,
                in_chans=3,
                embed_dim=self.embed_dim
            )
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # ------------------------------------------------------------------
        # 2) 将原始 ViT 的 pos_embed 拆分并插值成 template/search 各自的 pos_embed
        # ------------------------------------------------------------------

        # 取出 patch token 对应的位置编码（跳过 cls/distill token）
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]  # [1, num_patches, C]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)           # [1, C, num_patches]
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)      # [1, C, Ph, Pw]

        # search 区域 pos embed
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False
        )
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)  # [1, N_x, C]

        # template 区域 pos embed
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False
        )
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)  # [1, N_z, C]

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # cls token 的 pos embed（代码中保留但通常不用于跟踪）
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # segment embedding（区分 template/search token）
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # 如果需要返回中间 stage 的特征，则为对应 stage 注册额外的 norm
        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        # ------------------------------------------------------------------
        # 3) 初始化选择模块 M
        #    input_dim = token 数 N = N_z + N_x
        #    output_dim = 候选后续层数 K = 12 - 1 - start_layer  (这里假设 depth=12)
        # ------------------------------------------------------------------
        self.MLP = ThreeLayerMLP(input_dim=320, output_dim=12 - 1 - start_layer)

    # --------------------------------------------------------------------------
    # 训练时的 forward：除了执行选中的后续层，还会计算 cos_tensor 用于监督 MLP
    # --------------------------------------------------------------------------
    def forward_(self, z: torch.Tensor, x: torch.Tensor):
        """训练前向（会返回 cos_tensor + pro 用于 similarity loss）

        输入
        ----
        z : torch.Tensor
            template 图像，[B, 3, H_z, W_z]
        x : torch.Tensor
            search 图像，[B, 3, H_x, W_x]

        返回
        ----
        out : torch.Tensor
            融合后的 token 特征，[B, N, C]
        aux_dict : dict
            - cos_tensor: [B, K]，每个候选后续层与饱和层特征的相似度（用来产生 one-hot 监督）
            - pro: [B, K]，MLP 输出的选择概率
        """

        B = x.shape[0]

        # 1) patch embedding：图像 -> token 序列
        z = self.patch_embed(z)  # [B, N_z, C]
        x = self.patch_embed(x)  # [B, N_x, C]

        # 2) 加上位置编码
        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        # 3) one-stream：拼接 template/search token
        x = combine_tokens(z, x, mode=self.cat_mode)  # [B, N_z+N_x, C]
        x = self.pos_drop(x)

        # 4) 跑 transformer blocks，并在饱和层处启动“动态层选择”
        for i, blk in enumerate(self.blocks):
            if i < start_layer:
                # 饱和层之前：正常执行
                x = blk(x)

            elif i == start_layer:
                # 走到饱和层：执行该层并缓存输出作为 mid = X^{l*}
                x = blk(x)
                mid = x.detach()  # 作为相似度目标的输入，不回传梯度

                # 选择模块输入：取第 0 个通道在 token 维度的向量 => [B, N]
                # （论文中的 e_1^T X^{l*}）
                pro = self.MLP(x[:, :, 0].clone())  # [B, K]

                # 选出 top-k（默认 1）个候选后续层的索引
                # topk_indices 的范围是 [0, K-1]，因此需要 + (start_layer+1) 映射回真实 layer id
                _, topk_indices = torch.topk(pro, enabled_layer_num, dim=1)
                sorted_topk_indices = torch.sort(topk_indices, dim=1).values + start_layer + 1

            else:
                # 饱和层之后：仅对“选择到该层”的样本执行该层
                # sorted_topk_indices: [B, enabled_layer_num]
                idx = torch.where(sorted_topk_indices[:, :] == i)[0]
                if len(idx) > 0:
                    x[idx] = blk(x[idx])

        # 5) 构造 similarity 监督信号：计算每个候选后续层与 mid 的余弦相似度
        #    - 注意：这里对每个候选层 i，都单独计算 temp = blk(mid)
        #      这对应“跳过中间层，直接把 mid 喂给某个后续层”的使用方式。
        with torch.no_grad():
            cos_tensor = torch.ones(B, 12 - 1 - start_layer, device=x.device)
            for i, blk in enumerate(self.blocks):
                if i > start_layer:
                    temp = blk(mid)  # mid 直接过候选层 i
                    # F.cosine_similarity 默认 dim=1，这里 mid/temp 形状 [B, N, C]
                    # => 得到 [B, C]，再对 C 求均值得到每个样本一个相似度标量
                    cos = F.cosine_similarity(mid, temp)
                    cos_tensor[:, i - (start_layer + 1)] = cos.mean(dim=1)

        # 6) 恢复 token 顺序（某些 cat_mode 会改变顺序），并做最后的 LayerNorm
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        # aux_dict 会被上层网络/actor 用来计算 similarity loss
        aux_dict = {
            "attn": None,
            "cos_tensor": cos_tensor.detach(),
            "pro": pro,
        }

        return self.norm(x), aux_dict

    def forward(self, z, x, **kwargs):
        """统一 forward：训练时走 forward_，推理时走 forward_test"""
        if self.training:
            x, aux_dict = self.forward_(z, x)
        else:
            x, aux_dict = self.forward_test(z, x)
        return x, aux_dict

    # --------------------------------------------------------------------------
    # 测试/跟踪时的 forward：只需要执行选中的层，不需要计算 cos_tensor
    # --------------------------------------------------------------------------
    def forward_test(self, z: torch.Tensor, x: torch.Tensor):
        """推理前向（跟踪时通常 batch=1，因此实现里可以 break）"""

        B = x.shape[0]

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < start_layer:
                x = blk(x)

            elif i == start_layer:
                x = blk(x)
                pro = self.MLP(x[:, :, 0].clone())
                _, topk_indices = torch.topk(pro, enabled_layer_num, dim=1)
                sorted_topk_indices = torch.sort(topk_indices, dim=1).values + start_layer + 1

            else:
                idx = torch.where(sorted_topk_indices[:, :] == i)[0]
                if len(idx) > 0:
                    # 对选择到该层的样本执行该层
                    x[idx] = blk(x[idx])

                    # 注意：这里 break 的前提是 B=1（逐帧跟踪）。
                    # 若 B>1 且不同样本选择的层不同，这里会提前结束导致部分样本未执行选中层。
                    break

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        aux_dict = {"attn": None}

        return self.norm(x), aux_dict
