from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
from .until_module import LayerNorm, ACT2FN
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model=512,               # 特征维度
        nhead=8,                   # 多头注意力头数
        num_layers=4,              # Transformer 堆叠层数
        dim_feedforward=2048,      # 前馈层维度
        dropout=0.1,               # Dropout 概率
        num_visual_expansion_tokens=2  # 视觉扩展 token 个数
    ):  
        super().__init__()
        
        self.d_model = d_model
        self.num_visual_expansion_tokens = num_visual_expansion_tokens
        
        # ===== 可学习的视觉扩展 token（增强视频全局表示能力） =====
        # 形状: [1, num_visual_expansion_tokens, d_model]
        self.visual_expansion_tokens = nn.Parameter(
            torch.zeros(1, num_visual_expansion_tokens, d_model)
        )
        nn.init.normal_(self.visual_expansion_tokens, std=0.02)  # 正态分布初始化
        
        # ===== 时间维度上的 Transformer 编码器 =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 输入形状 [batch, seq, dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, frame_features):
        """
        输入:
            frame_features: [batch_size, num_frames, d_model]  视频帧特征
        输出:
            video_features: [batch_size, num_frames + num_visual_expansion_tokens, d_model]
        """
        batch_size = frame_features.shape[0]
        
        # 将视觉扩展 token 扩展到 batch 维度
        expansion_tokens = self.visual_expansion_tokens.expand(batch_size, -1, -1)
        
        # 将帧特征和扩展 token 拼接
        combined_features = torch.cat([frame_features, expansion_tokens], dim=1)
        
        # 输入 Transformer，进行时序建模
        video_features = self.transformer(combined_features)
        
        return video_features
