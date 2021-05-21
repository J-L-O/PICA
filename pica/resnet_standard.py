#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from lib import Config as cfg
from lib.networks import DefaultModel, Flatten, register
from lib.utils.loggers import STDLogger as logger

__all__ = ['ResNet34Standard']

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet34Standard(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')

    def __init__(self, cin, cout, sobel, grayscale, net_heads=None):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(ResNet34Standard, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        self.grayscale = self._make_grayscale_() if grayscale else None
        # build trunk net
        self.inplanes = 64

        if sobel:
            input_channels = 2
        elif grayscale:
            input_channels = 1
        else:
            input_channels = cin

        self.layer1 = nn.Sequential(nn.Conv2d(input_channels, 64,
                    kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = self._make_layer(BasicBlock, 64, 3)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())
        heads = [nn.Sequential(nn.Linear(512 * BasicBlock.expansion, head),
            nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion,
                        track_running_stats=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                                    track_running_stats=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                    track_running_stats=True))

        return nn.Sequential(*layers)

    def forward(self, x, hidx, target=7):
        if target is None or target > 7:
            raise NotImplementedError('Target is expected to be smaller than 8')

        # for i in range(x.shape[0]):
        #     plt.imshow(x[i].permute(1, 2, 0).cpu())
        #     plt.savefig(f'{i}_color.png')
        #     plt.close()

        if self.sobel is not None:
            x = self.sobel(x)
        elif self.grayscale is not None:
            x = self.grayscale(x)

            # combined_sobel = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).cpu()
            # for i in range(combined_sobel.shape[0]):
            #     plt.imshow(combined_sobel[i], cmap='gray')
            #     plt.savefig(f'{i}_sobel.png')
            #     plt.close()

        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.avgpool, self.heads[hidx]]
        for layer in layers[:target]:
            x = layer(x)
        return x

register('resnet34standard', ResNet34Standard)
