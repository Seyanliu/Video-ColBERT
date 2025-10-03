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
        num_visual_expansion_tokens=2,  # 视觉扩展 token 个数
        pretrained_text_state_dict=None
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

        if pretrained_text_state_dict is not None:
            self._init_from_clip_text_encoder(pretrained_text_state_dict, num_layers)
        
    def _init_from_clip_text_encoder(self, clip_text_state_dict, num_layers):
        """从CLIP文本编码器加载前num_layers层的参数"""
        # 遍历时序Transformer的每一层
        for layer_idx in range(num_layers):
            # CLIP文本Transformer层的参数前缀 (resblocks.{i})
            clip_layer_prefix = f"transformer.resblocks.{layer_idx}"
            # 本模型Transformer层的参数前缀 (layers.{i})
            target_layer_prefix = f"transformer.layers.{layer_idx}"
            
            # 加载多头注意力层参数
            # 注意力查询/键/值投影权重
            attn_in_proj_weight = clip_text_state_dict[f"{clip_layer_prefix}.attn.in_proj_weight"]
            self.state_dict()[f"{target_layer_prefix}.self_attn.in_proj_weight"].copy_(attn_in_proj_weight)
            
            # 注意力查询/键/值投影偏置
            attn_in_proj_bias = clip_text_state_dict[f"{clip_layer_prefix}.attn.in_proj_bias"]
            self.state_dict()[f"{target_layer_prefix}.self_attn.in_proj_bias"].copy_(attn_in_proj_bias)
            
            # 注意力输出投影权重
            attn_out_proj_weight = clip_text_state_dict[f"{clip_layer_prefix}.attn.out_proj.weight"]
            self.state_dict()[f"{target_layer_prefix}.self_attn.out_proj.weight"].copy_(attn_out_proj_weight)
            
            # 注意力输出投影偏置
            attn_out_proj_bias = clip_text_state_dict[f"{clip_layer_prefix}.attn.out_proj.bias"]
            self.state_dict()[f"{target_layer_prefix}.self_attn.out_proj.bias"].copy_(attn_out_proj_bias)
            
            # 加载前馈网络参数
            # 第一层线性变换权重
            mlp_c_fc_weight = clip_text_state_dict[f"{clip_layer_prefix}.mlp.c_fc.weight"]
            self.state_dict()[f"{target_layer_prefix}.linear1.weight"].copy_(mlp_c_fc_weight)
            
            # 第一层线性变换偏置
            mlp_c_fc_bias = clip_text_state_dict[f"{clip_layer_prefix}.mlp.c_fc.bias"]
            self.state_dict()[f"{target_layer_prefix}.linear1.bias"].copy_(mlp_c_fc_bias)
            
            # 第二层线性变换权重
            mlp_c_proj_weight = clip_text_state_dict[f"{clip_layer_prefix}.mlp.c_proj.weight"]
            self.state_dict()[f"{target_layer_prefix}.linear2.weight"].copy_(mlp_c_proj_weight)
            
            # 第二层线性变换偏置
            mlp_c_proj_bias = clip_text_state_dict[f"{clip_layer_prefix}.mlp.c_proj.bias"]
            self.state_dict()[f"{target_layer_prefix}.linear2.bias"].copy_(mlp_c_proj_bias)
            
            # 加载层归一化参数
            ln_1_weight = clip_text_state_dict[f"{clip_layer_prefix}.ln_1.weight"]
            self.state_dict()[f"{target_layer_prefix}.norm1.weight"].copy_(ln_1_weight)
            
            ln_1_bias = clip_text_state_dict[f"{clip_layer_prefix}.ln_1.bias"]
            self.state_dict()[f"{target_layer_prefix}.norm1.bias"].copy_(ln_1_bias)
            
            ln_2_weight = clip_text_state_dict[f"{clip_layer_prefix}.ln_2.weight"]
            self.state_dict()[f"{target_layer_prefix}.norm2.weight"].copy_(ln_2_weight)
            
            ln_2_bias = clip_text_state_dict[f"{clip_layer_prefix}.ln_2.bias"]
            self.state_dict()[f"{target_layer_prefix}.norm2.bias"].copy_(ln_2_bias)

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
