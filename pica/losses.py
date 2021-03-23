#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import Config as cfg


class ContinuousCrossEntropyLoss(nn.Module):

    def forward(self, x, y):
        return torch.sum(-y * x.log())


class PUILoss(nn.Module):

    def __init__(self, lamda=2.0, target=None):
        super(PUILoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.target = torch.FloatTensor(target).to(cfg.device)
        self.cce = ContinuousCrossEntropyLoss()

    def forward(self, x, y):
        """Partition Uncertainty Index
        
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]
        
        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == y.shape, 'Inputs are required to have same shape'

        # partition uncertainty index
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.shape[0]).to(cfg.device))

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()

        if self.target is None or x.shape[1] != len(self.target):
            loss_ne = math.log(p.shape[0]) + (p * p.log()).sum()
        else:
            loss_ne = self.cce(p, self.target)

        return loss_ce + self.lamda * loss_ne
