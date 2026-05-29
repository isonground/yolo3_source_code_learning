# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils.patches import torch_load

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Computes the weighted or unweighted sum of multiple input layers per https://arxiv.org/abs/1911.09070."""
    """计算多个输入层的加权或非加权和，参考论文 https://arxiv.org/abs/1911.09070。"""

    def __init__(self, n, weight=False):  # n: number of inputs
        """Initializes a module to compute weighted/unweighted sum of n inputs, with optional learning weights.

        https://arxiv.org/abs/1911.09070
        """
        """初始化一个模块，用于计算 n 个输入的加权/非加权和，并可选择学习权重。

        论文链接：https://arxiv.org/abs/1911.09070
        参数:
            n (int): 输入张量的个数
            weight (bool): 是否使用可学习的加权和，默认 False 表示直接求和
        """
        super().__init__()
        # 是否应用权重标志
        self.weight = weight  # apply weights boolean
        # 迭代对象，用于遍历除第一个输入之外的其余输入
        self.iter = range(n - 1)  # iter object
        if weight:
            # 可学习的权重参数，初始值为负的等差数列，经过 sigmoid 变换后会限制在 [0,2] 范围内
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Performs forward pass, blending `x` elements with optional learnable weights.

        See https://arxiv.org/abs/1911.09070 for more.
        """
        """执行前向传播，将 x 中的元素加权或直接相加。

        论文参考：https://arxiv.org/abs/1911.09070
        参数:
            x (list of torch.Tensor): 输入张量列表
        返回:
            torch.Tensor: 加权或非加权求和后的张量
        """
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Implements mixed depth-wise convolutions for efficient neural networks; see https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """Initializes MixConv2d with mixed depth-wise convolution layers; details at https://arxiv.org/abs/1907.09595.
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Applies a series of convolutions, batch normalization, and SiLU activation to input tensor `x`."""
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Combines outputs from multiple models to improve inference results."""

    def __init__(self):
        """Initializes an ensemble of models to combine their outputs."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Applies ensemble of models on input `x`, with options for augmentation, profiling, and visualization,
        returning inference outputs.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """Loads an ensemble or single model weights, supports device placement and model fusion."""
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch_load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
