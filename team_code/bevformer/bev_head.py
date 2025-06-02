import copy
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
import numpy as np
import mmcv
import cv2 as cv
from .transformer import PerceptionTransformer
from mmdet.models.utils import build_transformer

@HEADS.register_module()
class BEVHead(BaseModule):
    def __init__(self, 
                 bev_h,
                 bev_w,
                 pc_range,
                 embed_dims,
                 transformer, 
                 positional_encoding: dict,
                 init_cfg=None,
                 **kwargs,
                 ):
        super(BEVHead, self).__init__(init_cfg=init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.transformer :PerceptionTransformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        
        self._init_layers()
    def init_weights(self):
        """Initialize weights of the Multi View BEV Encoder"""
        self.transformer.init_weights()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    @force_fp32(apply_to=('mlvl_feats', 'pred_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        if only_bev:
            return bev_embed
        
        bev_feature = bev_embed.permute(0, 2, 1).reshape(bs, self.embed_dims, self.bev_h, self.bev_w)
        return bev_feature 