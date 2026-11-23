"""无预训练权重的 DeepLabV3+（MobileNetV2-0.5 编码器，output_stride=16）。

本模块提供用于 128x128 RGB 建筑物提取的轻量级 DeepLabV3+ 纯 PyTorch 实现：
- 编码器：MobileNetV2（width_mult=0.5，output_stride=16，阶段 6/7 空洞率 dilation=2）
- 空间金字塔池化：ASPP（空洞率 6, 12, 18 + 全局平均池化分支，输出 128 通道）
- 解码器：低层特征（32x32, 16 通道投影至 32 通道）与 ASPP 特征融合，初始 128 通道轻量解码器
- 输出：单通道双线性上采样至输入分辨率并经过 Sigmoid 激活的概率图
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """将通道数对齐为指定因子（默认 8）的整数倍，保持与 MobileNetV2 规范一致。"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """标准 Conv2d + BatchNorm2d + ReLU/ReLU6 基础卷积单元。"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: type[nn.Module] = nn.ReLU6,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_planes,
                            out_planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_planes)),
                    ("act", activation(inplace=True)),
                ]
            )
        )


class InvertedResidual(nn.Module):
    """支持空洞卷积的 MobileNetV2 倒残差瓶颈块。"""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"InvertedResidual stride 仅支持 1 或 2，收到 {stride}")
        self.stride = stride
        self.dilation = dilation
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(round(inp * expand_ratio))
        layers: list[tuple[str, nn.Module]] = []
        if expand_ratio != 1:
            # 1x1 逐点升维卷积
            layers.append(("expand", ConvBNReLU(inp, hidden_dim, kernel_size=1)))
        # 3x3 深度可分离卷积（支持空洞率控制输出步幅）
        layers.append(
            (
                "depthwise",
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    groups=hidden_dim,
                ),
            )
        )
        # 1x1 线性投影（无激活函数）
        layers.append(("project_conv", nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False)))
        layers.append(("project_bn", nn.BatchNorm2d(oup)))
        self.conv = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 编码器（width_mult=0.5，output_stride=16），同时返回低层与高层特征。"""

    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 0.5,
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        if output_stride != 16:
            raise ValueError(f"当前实验锁定 output_stride=16，收到 {output_stride}")
        self.width_mult = float(width_mult)
        self.output_stride = int(output_stride)

        input_channel = _make_divisible(32 * self.width_mult, 8)
        self.stem = ConvBNReLU(in_channels, input_channel, kernel_size=3, stride=2)

        # (expand_ratio, base_channels, num_blocks, nominal_stride)
        inverted_residual_setting = [
            (1, 16, 1, 1),   # stage 1: stride 2 -> 64x64
            (6, 24, 2, 2),   # stage 2: stride 4 -> 32x32（低层特征提取点）
            (6, 32, 3, 2),   # stage 3: stride 8 -> 16x16
            (6, 64, 4, 2),   # stage 4: stride 16 -> 8x8
            (6, 96, 3, 1),   # stage 5: stride 16 -> 8x8
            (6, 160, 3, 2),  # stage 6: 原 stride 2 调整为 stride=1, dilation=2 -> 8x8
            (6, 320, 1, 1),  # stage 7: dilation=2 -> 8x8
        ]

        current_stride = 2
        current_dilation = 1
        self.low_level_stage_index = 1  # stage 2（0-based index=1）
        self.low_level_channels = 0
        self.stages = nn.ModuleList()

        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * self.width_mult, 8)
            blocks: list[nn.Module] = []
            for block_idx in range(n):
                if block_idx == 0:
                    if current_stride == self.output_stride:
                        stride = 1
                        dilation = current_dilation * s
                        current_dilation = dilation
                    else:
                        stride = s
                        dilation = current_dilation
                        current_stride *= s
                else:
                    stride = 1
                    dilation = current_dilation
                blocks.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                        dilation=dilation,
                    )
                )
                input_channel = output_channel
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx == self.low_level_stage_index:
                self.low_level_channels = output_channel

        self.high_level_channels = input_channel

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (low_level_features [stride=4], high_level_features [stride=16])。"""
        out = self.stem(x)
        low_level: torch.Tensor | None = None
        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx == self.low_level_stage_index:
                low_level = out
        assert low_level is not None
        return low_level, out


class ASPPPooling(nn.Module):
    """ASPP 全局平均池化分支：先插值恢复空间尺寸再做 BatchNorm，兼容单样本 batch=1 训练。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feat = self.pool(x)
        feat = self.conv(feat)
        feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
        feat = self.bn(feat)
        return self.act(feat)


class ASPP(nn.Module):
    """空洞空间金字塔池化模块（rates=(6, 12, 18)）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            ConvBNReLU(in_channels, out_channels, kernel_size=1, activation=nn.ReLU)
        ]
        for rate in atrous_rates:
            modules.append(
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=rate,
                    activation=nn.ReLU,
                )
            )
        modules.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(modules)
        self.project = ConvBNReLU(
            len(self.branches) * out_channels,
            out_channels,
            kernel_size=1,
            activation=nn.ReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [branch(x) for branch in self.branches]
        return self.project(torch.cat(res, dim=1))


class DeepLabV3Plus(nn.Module):
    """面向 128x128 遥感建筑物二分类的轻量 DeepLabV3+ 模型（随机初始化，无预训练）。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width_mult: float = 0.5,
        output_stride: int = 16,
        aspp_rates: tuple[int, int, int] = (6, 12, 18),
        low_level_proj_channels: int = 32,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width_mult = float(width_mult)
        self.output_stride = int(output_stride)
        self.aspp_rates = tuple(int(r) for r in aspp_rates)
        self.low_level_proj_channels = int(low_level_proj_channels)
        self.decoder_channels = int(decoder_channels)

        self.encoder = MobileNetV2Backbone(
            in_channels=self.in_channels,
            width_mult=self.width_mult,
            output_stride=self.output_stride,
        )
        self.aspp = ASPP(
            in_channels=self.encoder.high_level_channels,
            out_channels=self.decoder_channels,
            atrous_rates=self.aspp_rates,
        )
        self.low_level_proj = ConvBNReLU(
            self.encoder.low_level_channels,
            self.low_level_proj_channels,
            kernel_size=1,
            activation=nn.ReLU,
        )
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec_conv1",
                        ConvBNReLU(
                            self.decoder_channels + self.low_level_proj_channels,
                            self.decoder_channels,
                            kernel_size=3,
                            activation=nn.ReLU,
                        ),
                    ),
                    (
                        "dec_conv2",
                        ConvBNReLU(
                            self.decoder_channels,
                            self.decoder_channels,
                            kernel_size=3,
                            activation=nn.ReLU,
                        ),
                    ),
                    (
                        "classifier",
                        nn.Conv2d(self.decoder_channels, self.out_channels, kernel_size=1),
                    ),
                ]
            )
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """使用确定性 Kaiming/常数初始化，不加载任何外部预训练权重。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入 (B, 3, H, W)，输出 (B, 1, H, W) 的 [0, 1] 概率图。"""
        input_size = x.shape[-2:]
        low_level, high_level = self.encoder(x)
        aspp_feat = self.aspp(high_level)
        aspp_up = F.interpolate(
            aspp_feat,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_proj = self.low_level_proj(low_level)
        fused = torch.cat([aspp_up, low_proj], dim=1)
        logits = self.decoder(fused)
        logits_up = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits_up)


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """统计模型总参数量、可训练参数量以及主要子模块参数明细。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    submodules: dict[str, int] = {}
    for name, child in model.named_children():
        submodules[name] = sum(p.numel() for p in child.parameters())
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "buffer_elements": int(buffers),
        "submodule_parameters": submodules,
    }


def architecture_summary(model: nn.Module) -> dict[str, Any]:
    """记录模型结构超参数与各层输出摘要，用于实验归档与对比。"""
    stats = count_parameters(model)
    summary: dict[str, Any] = {
        "class_name": model.__class__.__name__,
        **stats,
        "repr": str(model),
    }
    if isinstance(model, DeepLabV3Plus):
        summary["spec"] = {
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "encoder": "MobileNetV2",
            "width_mult": model.width_mult,
            "output_stride": model.output_stride,
            "low_level_raw_channels": model.encoder.low_level_channels,
            "high_level_channels": model.encoder.high_level_channels,
            "aspp_rates": list(model.aspp_rates),
            "low_level_proj_channels": model.low_level_proj_channels,
            "decoder_channels": model.decoder_channels,
            "pretrained": False,
        }
    return summary
