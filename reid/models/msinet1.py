import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import *
from .sam import AlignModule
from reid.utils.serialization import copy_state_dict


class Cell(nn.Module):
    """Basic form of a cell.
    Consisted of two branches, with 2 and 6 light conv 3x3, respectively.
    There are two interaction modules in the middle and tail of the cell."""

    def __init__(self, in_channels, out_channels, genotypes):
        super(Cell, self).__init__()
        mid_channels = out_channels // 4
        # 第一个分支 1x1卷积和3x3轻量级卷积
        self.conv1a = Conv1x1(in_channels, mid_channels)
        self.conv1b = Conv1x1(in_channels, mid_channels)
        # 第二个分支包含1x1卷积和3个3x3轻量级卷积。
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # 第一个交互模块.
        self._op2 = OPS[genotypes[0]](mid_channels, mid_channels)

        self.conv3a = LightConv3x3(mid_channels, mid_channels)
        self.conv3b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # 第二个交互模块.
        self._op3 = OPS[genotypes[1]](mid_channels, mid_channels)

        self.conv4a = LightConv3x3(mid_channels, mid_channels)
        self.conv4b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # Create interaction options.
        self._op4 = OPS[genotypes[2]](mid_channels, mid_channels)

        # conv4a、conv4b将两个分支的输出通过1x1卷积线性融合.
        self.conv5a = Conv1x1Linear(mid_channels, out_channels)
        self.conv5b = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1a = self.conv1a(x)
        x1b = self.conv1b(x)

        x2a = self.conv2a(x1a)
        x2b = self.conv2b(x1b)
        x2a, x2b = self._op2((x2a, x2b))

        x3a = self.conv3a(x2a)
        x3b = self.conv3b(x2b)
        x3a, x3b = self._op3((x3a, x3b))

        x4a = self.conv3a(x3a)
        x4b = self.conv3b(x3b)
        x4a, x4b = self._op4((x4a, x4b))

        x4 = self.conv5a(x4a) + self.conv5b(x4b)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x4 + identity
        return F.relu(out)


class MSINet(nn.Module):
    """The basic structure of the proposed MSINet."""

    def __init__(self, args, num_classes, channels, genotypes):
        super(MSINet, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        # 输入层：7x7卷积和最大池化。
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # 多个 Cell 组成的中间层。
        self.cells = nn.ModuleList()
        # Consisted of 6 cells in total.
        for i in range(3):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            print(genotypes[i * 4 : i * 4 + 4])
            self.cells += [
                Cell(in_channels, out_channels, genotypes[i * 6 : i * 6 + 3]),
                Cell(out_channels, out_channels, genotypes[i * 6 + 3 : i * 6 + 6])
            ]
            if i != 2:
                # Downsample
                self.cells += [
                    nn.Sequential(
                        Conv1x1(out_channels, out_channels),
                        nn.AvgPool2d(2, stride=2)
                    )
                ]
        # GAP层（全局平均池化）：对最后一个 Cell 输出进行全局平均池化。
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.BatchNorm1d(channels[-1])
        self.head.bias.requires_grad_(False)
        # 多个全连接层：包括分类层和用于特征的降维层。
        self.fc = nn.Linear(channels[-1], channels[-1], bias=False)
        self.classifier = nn.Linear(channels[-1], num_classes)

        self.f_conv = nn.Conv2d(channels[-2], 128, 1)
        self.f_head = nn.BatchNorm1d(256)
        self.f_head.bias.requires_grad_(False)
        self.f_fc = nn.Linear(256, 256, bias=False)
        self.f_classifier = nn.Linear(256, num_classes)
        # 可选的SAM模块：根据配置使用或不使用SAM（Spatial Attention Module）模块。
        self.sam_mode = args.sam_mode
        if args.sam_mode != 'none':
            self.align_module = AlignModule(16, 8, channels[-1])

        self._init_params()

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for cell_idx, cell in enumerate(self.cells):
            x = cell(x)
            if cell_idx == 5:
                f_x = x

        return x, f_x

    def forward(self, x, train_transfer=False, test_transfer=False):
        x, f_x = self.featuremaps(x)

        if self.sam_mode != 'none':
            sam_scores = self.align_module(x)
        else:
            sam_scores = None

        v = self.gap(x).view(x.shape[0], -1)
        f_x = self.f_conv(f_x)
        height = f_x.shape[2]
        f_v_up = self.gap(f_x[:, :, :height // 2, :]).view(f_x.shape[0], -1)
        f_v_down = self.gap(f_x[:, :, height // 2:, :]).view(f_x.shape[0], -1)
        f_v = torch.cat((f_v_up, f_v_down), dim=1)

        n_v = self.head(self.fc(v))
        n_f_v = self.f_head(self.f_fc(f_v))

        if not self.training:
            if test_transfer:
                return torch.cat((n_v, n_f_v), dim=1)
            else:
                return torch.cat((v, f_v), dim=1)

        y = self.classifier(n_v)
        f_y = self.f_classifier(n_f_v)

        if train_transfer:
            return n_v, y, n_f_v, f_y, sam_scores
        else:
            return v, y, f_v, f_y, sam_scores

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def msinet_x1_0(args, num_classes=1000):
    genotypes, pretrained_weight = genotype_factory[args.genotypes]
    model = MSINet(
        args,
        num_classes,
        channels=[64, 256, 384, 512],
        genotypes=genotypes
    )
    if args.pretrained:
        copy_state_dict(
            torch.load(
                osp.join(args.pretrain_dir, pretrained_weight)
            )['state_dict'], model
        )
    return model
