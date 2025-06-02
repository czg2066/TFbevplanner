import copy
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule

@HEADS.register_module()
class BEVFormerHead(BaseModule):
    """Head of Detr3D.
    Args:
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 as_two_stage=False,
                 transformer_dict=None,
                 embed_dims=256,
                 num_query=80,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):
        super(BEVFormerHead, self).__init__(init_cfg=None)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer_dict['as_two_stage'] = self.as_two_stage
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims/2,
            normalize=True)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer_dict)

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas=None, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        return self.transformer(
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