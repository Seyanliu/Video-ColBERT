import os

import torch
import torch.nn.functional as F
from torch import nn

from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2

allgather = AllGather.apply
allgather2 = AllGather2.apply


class VideoColBERTLoss(nn.Module):
    def __init__(self, temperature=None, bias=None, lambda_f=1.0, lambda_v=1.0):
        super().__init__()
        
        # 温度参数设置
        if temperature is not None:
            self.temperature = nn.Parameter(torch.tensor([temperature]))  # 共享模型的温度参数
            
        # 偏置参数设置  
        if bias is not None:
            self.bias = nn.Parameter(torch.tensor([bias]))  # 共享模型的偏置参数
            
        # 损失权重超参数
        self.lambda_f = lambda_f  # 帧级损失权重 λ_F
        self.lambda_v = lambda_v  # 视频级损失权重 λ_V
    def forward(self, mms_f, mms_v):

        B = mms_f.size(0)

        # 构造标签矩阵 z: [B, B], 对角线=1, 其余=-1
        z = torch.ones_like(mms_f)
        z.fill_(-1.0)
        z.fill_diagonal_(1.0)
        # print("z shape:", z.shape)   

        # L_F
        logits_f = -self.temperature * mms_f + self.bias   # [B, B]
        logits_f = logits_f * -1.0
        loss_f = -torch.mean(F.logsigmoid(z * logits_f))

        # L_V
        logits_v = -self.temperature * mms_v + self.bias   # [B, B]
        logits_v = logits_v * -1.0
        loss_v = -torch.mean(F.logsigmoid(z * logits_v))

        loss = loss_f * self.lambda_f + loss_v * self.lambda_v

        return loss, loss_f, loss_v


class VideoColBERT(nn.Module):
    def __init__(self, config):
        super(VideoColBERT, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        print(f"Successfully loaded CLIP model")
        print(f"Feature dimension: {embed_dim}")

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        # 冻结文本模型的 token embedding 和 positional embedding
        for param in self.clip.token_embedding.parameters():
            param.requires_grad = False
        self.clip.positional_embedding.requires_grad = False

        # 冻结视觉模型的 patch embedding (conv1)、positional embedding 和 cls embedding
        for param in self.clip.visual.conv1.parameters():
            param.requires_grad = False
        self.clip.visual.positional_embedding.requires_grad = False
        self.clip.visual.class_embedding.requires_grad = False

        
        # VideoColBERT 相关参数
        self.use_query_expansion = getattr(config, 'use_query_expansion', True)
        self.query_pad_length = getattr(config, 'query_pad_length', 32)
        self.visual_expansion_tokens = getattr(config, 'visual_expansion_tokens', 2)
        self.temporal_layers = getattr(config, 'temporal_layers', 4)
        self.temporal_heads = getattr(config, 'temporal_heads', 8)
        self.temporal_ff_dim = getattr(config, 'temporal_ff_dim', 2048)
        self.temporal_dropout = getattr(config, 'temporal_dropout', 0.1)

        
        if self.agg_module == "seqTransf":
            self.transformerClip = TransformerClip(
                d_model=embed_dim,
                nhead=self.temporal_heads,
                num_layers=self.temporal_layers,
                dim_feedforward=self.temporal_ff_dim,
                dropout=self.temporal_dropout,
                num_visual_expansion_tokens=self.visual_expansion_tokens,
                pretrained_text_state_dict=state_dict  # 传入CLIP的状态字典
                num_visual_expansion_tokens=self.visual_expansion_tokens,
                pretrained_text_state_dict=state_dict  # 传入CLIP的状态字典
            )

        # 可学习参数：温度缩放 & 偏置 (根据原论文设置)
        # 原论文参数: t = 4.77, b = -12.93 (4.77太大会导致sigmoid饱和)
        self.temperature = nn.Parameter(torch.tensor([4.77]))
        self.bias = nn.Parameter(torch.tensor([-12.93]))

        # VideoColBERT 专用损失函数
        lambda_f = getattr(config, 'lambda_f', 1.0)
        lambda_v = getattr(config, 'lambda_v', 1.0)

        # 损失函数：交叉熵损失用于对比学习
        self.loss = VideoColBERTLoss(
            temperature=self.temperature, 
            bias=self.bias, 
            lambda_f=lambda_f, 
            lambda_v=lambda_v
        )

        # self.apply(self.init_weights)  # random init must before loading pretrain
        # self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

    def forward(self, text_ids, text_mask, video, video_mask=None):
        """
        模型前向传播函数
        
        参数:
            text_ids: 文本token的ID序列 [B, L_t]
            text_mask: 文本掩码，标识有效token [B, L_t] 
            video: 视频帧数据 [B, N_v, 3, H, W]
            video_mask: 视频掩码，标识有效帧 [B, N_v]
            
        返回:
            训练时返回损失值，推理时返回None
        """
        # 重塑输入张量维度
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # 视频数据：B x N_v x 3 x H x W -> (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        b, n_v, d, h, w = video.shape
        video = video.view(b * n_v, d, h, w)

        # 提取文本和视频特征
        text_feat, frame_feat, video_feat = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            # 训练模式：计算相似度矩阵和损失
            sim_matrix1, sim_matrix2, mms_f, mms_v = self.get_similarity_logits(text_feat, frame_feat, video_feat, shaped=True)
            # 总损失 = loss + λ * CDCR损失（跨域相关性正则化）
            loss, loss_f, loss_v= self.loss(mms_f, mms_v)

            return loss, loss_f, loss_v
        else:
            return None

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        """
        提取文本特征 

        参数:
            text_ids: 文本token ID序列 [batch_size, text_length]
            text_mask: 文本掩码 [batch_size, text_length]
            shaped: 是否已经重塑过维度

        返回:
            text_feat: 文本特征 [batch_size, query_pad_length, d_model]
        """
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        batch_size = text_ids.size(0)

        text_feat = self.clip.encode_text(text_ids, return_hidden=True)[1].float()   # [batch_size, query_pad_length, d_model]
        text_feat = text_feat.view(batch_size, -1, text_feat.size(-1)) # [batch_size, query_pad_length, d_model]

        return text_feat # [batch_size, query_pad_length, d_model]

    def get_video_feat(self, video, video_mask, shaped=False):
        """
        提取视频特征
        
        参数:
            video: 视频帧数据 [batch_size * num_frames, 3, H, W]
            video_mask: 视频掩码
            shaped: 是否已经重塑过维度
            
        返回:
            frame_feat: 帧特征   [batch_size, num_frames, d_model]
            video_feat: 视频特征 [batch_size, num_frames + num_expansion, d_model]
        """
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)

        batch_size = video_mask.size(0)
        num_frames = video_mask.size(1)
        num_frames = video_mask.size(1)
        image_feat = self.clip.encode_image(video).float()
        frame_feat = image_feat.float().view(batch_size, -1, image_feat.size(-1)) # [batch_size, num_frames, d_model]

        # video_mask = video_mask.unsqueeze(-1).float()  # [B, T, 1]，扩展维度以匹配frame_feat
        # frame_feat = frame_feat * video_mask  # [B, T, D]，无效帧特征置0

        video_feat = self.transformerClip(frame_feat)

        return frame_feat, video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        """
        同时提取文本和视频特征
        
        参数:
            text_ids, text_mask: 文本输入和掩码
            video, video_mask: 视频输入和掩码  
            shaped: 是否已经重塑过维度
            
        返回:
            text_feat, frame_feat, video_feat: 文本特征,帧特征和视频特征
        """
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            # ===== 视频 reshape =====
            # video: [B, N, 3, H, W] → [B*N, 3, H, W]
            video = torch.as_tensor(video).float()
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)

        text_feat = self.get_text_feat(text_ids, text_mask, shaped=True)
        frame_feat, video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, frame_feat, video_feat

    def mms_interaction(self, text_feat, frame_feat, video_feat):
        """
        MMS交互模式
        同时利用文本特征、帧特征和视频特征进行交互计算
        
        参数:
            text_feat: 文本特征 [batch_size, text_len, d_model]
            frame_feat: 帧特征  [batch_size, frame_len, d_model]
            video_feat: 视频特征 [batch_size, frames_len + expansion_len, d_model]
            
        返回:
            retrieve_logits: 文本到视频的相似度矩阵
            retrieve_logits.T: 视频到文本的相似度矩阵  
            cdcr_loss: 跨域相关性正则化损失
        """
        # 分布式训练时，进行 allgather，把不同 GPU 上的 batch 合并
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = allgather(text_feat.contiguous(), self.config)
            frame_feat = allgather(frame_feat.contiguous(), self.config)
            video_feat = allgather(video_feat.contiguous(), self.config)
            torch.distributed.barrier()  # force sync

        text_feat = text_feat.float()
        frame_feat = frame_feat.float()
        video_feat = video_feat.float()

        # 1) 计算 text 与 frame 和 video 的成对相似度： sim_tv -> [B, B, L_t, L_v]
        sim_tv = torch.einsum('bqd,cvd->bcqv', text_feat, video_feat)
        sim_tv = torch.einsum('bqd,cvd->bcqv', text_feat, video_feat)
        sim_tf = torch.einsum('bqd,cfd->bcqf', text_feat, frame_feat)

        # 2) 对 video 和 frame 维度取 max -> [B, B, L_t] 再取 mean -> [B, B]
        mms_v = sim_tv.max(dim=3).values.mean(dim=2)
        mms_f = sim_tf.max(dim=3).values.mean(dim=2)

        # 3) MMS_FV = MMS_F + MMS_V
        retrieve_logits =  mms_f + mms_v  # [B, B]
    
    
        return retrieve_logits, retrieve_logits.T, mms_f, mms_v


    def get_similarity_logits(self, text_feat, frame_feat, video_feat, shaped=False):
        """
        根据交互模式计算相似度logits
        
        参数:
            text_feat: 文本特征
            frame_feat: 帧特征
            video_feat: 视频特征
            shaped: 是否已经重塑过维度
            
        返回:
            t2v_logits: 文本到视频的相似度矩阵（应用温度缩放）
            v2t_logits: 视频到文本的相似度矩阵（应用温度缩放）  
            mms_f: 原始帧相似度分数
            mms_v: 原始视频相似度分数
        """
        # 根据配置的交互模式调用相应的交互函数
        if self.interaction == 'mms':
            t2v_logits, v2t_logits, mms_f, mms_v = self.mms_interaction(text_feat, frame_feat, video_feat)
        else:
            raise NotImplementedError
        
        return t2v_logits, v2t_logits, mms_f, mms_v

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
