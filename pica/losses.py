#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import math
import sys

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

        if target is not None:
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

        loss_iid_p = IIDLoss(x, y)

        # balance regularisation
        # p = x.sum(0).view(-1)
        # p /= p.sum()
        # if self.target is None or x.shape[1] != len(self.target):
        #     loss_ne = math.log(p.shape[0]) + (p * p.log()).sum()
        # else:
        #     loss_ne = self.cce(p, self.target)

        return loss_ce + loss_iid_p  # + self.lamda * loss_ne


def IIDLoss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.shape
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.shape == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS)] = EPS
    p_j[(p_j < EPS)] = EPS
    p_i[(p_i < EPS)] = EPS

    # loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    # loss = loss.sum()

    loss_no_lamb = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss_no_lamb


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.shape
    assert (x_tf_out.shape[0] == bn and x_tf_out.shape[1] == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j
