# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class FPNPixelDecoder(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    Modified from models.sgmentation.MaskHeadSmallConv
    """

    def __init__(self, conv_dim, mask_dim):
        super().__init__()
        self.mask_dim = mask_dim

        inter_dims = [2048, 1024, 512, 256]
        self.conv0 = nn.Sequential(nn.Conv2d(inter_dims[0], conv_dim, 3, padding=1),
                                   nn.GroupNorm(8, conv_dim),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
                                   nn.GroupNorm(8, conv_dim),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
                                   nn.GroupNorm(8, conv_dim),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
                                   nn.GroupNorm(8, conv_dim),
                                   nn.ReLU())
        
        self.adapter1 = nn.Sequential(nn.Conv2d(inter_dims[1], conv_dim, 1),
                                      nn.GroupNorm(8, conv_dim))
        self.adapter2 = nn.Sequential(nn.Conv2d(inter_dims[2], conv_dim, 1),
                                      nn.GroupNorm(8, conv_dim))
        self.adapter3 = nn.Sequential(nn.Conv2d(inter_dims[3], conv_dim, 1),
                                      nn.GroupNorm(8, conv_dim))

        self.conv_mask = nn.Conv2d(conv_dim, mask_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[NestedTensor]):
        # arange as coarse -> fine order
        feats = [ns.decompose()[0] for ns in features]
        feats = feats[::-1]

        x = self.conv0(feats[0])

        f = self.adapter1(feats[1])
        x = f + F.interpolate(x, size=f.shape[-2:], mode='nearest')
        x = self.conv1(x)
        
        f = self.adapter2(feats[2])
        x = f + F.interpolate(x, size=f.shape[-2:], mode='nearest')
        x = self.conv2(x)

        f = self.adapter3(feats[3])
        x = f + F.interpolate(x, size=f.shape[-2:], mode='nearest')
        x = self.conv3(x)

        x = self.conv_mask(x)
        return x


def build_pixel_decoder(args):
    decoder = FPNPixelDecoder(
        conv_dim=args.decode_conv_dim,
        mask_dim=args.mask_dim
    )
    return decoder
