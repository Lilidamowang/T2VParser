# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel
from mmengine.structures import InstanceData, BaseDataElement
from mmengine.runner import autocast

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, OptSampleList
from mmaction.models.losses import sim_matrix
from mmaction.datasets.transforms.text_transforms import tokenize



class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out

class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1, *args, **kwargs):
        super(FFN, self).__init__()
        self.ffn_layers = nn.ModuleList()
        self.ffn_layers.append(nn.Linear(dim, dim*4))
        self.ffn_layers.append(nn.ReLU())
        self.ffn_layers.append(nn.Linear(dim*4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.ffn_layers:
            x = layer(x)
            x = self.dropout(x)
        return x

class DisBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super(DisBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
       
        # Create a list of layers
        self.self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        # Add the attn, cross attention, simple feedforward layers to the list
        for _ in range(depth):
            # Add the multi query attention layer
            self.self_attn_layers.append(
                nn.MultiheadAttention(dim, heads, *args, **kwargs)
            )
            # Add the cross attention layer
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    dim,
                    heads,
                    dropout=dropout,
                    *args,
                    **kwargs,
                )
            )
            # Add the simple feedforward layer
            self.ffn_layers.append(
                FFN(dim, dropout)
            )

    def forward(self, MultiViewQuery: Tensor, info: Tensor) -> Tensor:
        num_Q = MultiViewQuery.shape[1]
        if len(info.shape) == 2:
            info = info.unsqueeze(1).expand(-1, num_Q, -1) # [16, 32, 512]
        if len(info.shape) == 3:
            MultiViewQuery = MultiViewQuery.permute(1, 0, 2)  # 将维度从(16, 32, 512)调整为(32, 16, 512)
            info = info.permute(1, 0, 2)  # 将维度从(16, 8, 512)调整为(8, 16, 512)
        for self_attn, cross_attn, ffn in zip(
            self.self_attn_layers,
            self.cross_attn_layers,
            self.ffn_layers,
        ):
            MultiViewQuery, _ = self_attn(MultiViewQuery,MultiViewQuery,MultiViewQuery) # 自注意力
            MultiViewQuery, _ = cross_attn(MultiViewQuery, info, info) # 交叉注意力
            MultiViewQuery = ffn(MultiViewQuery) # 前馈网络
        if len(info.shape) == 3:
            MultiViewQuery = MultiViewQuery.permute(1, 0, 2)
        return MultiViewQuery

@MODELS.register_module()
class MultiViewDis(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        num_query_token:int = 8,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        print("===MultiViewDis===")
        super(MultiViewDis, self).__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        
        self.visionBlock = DisBlock(dim, depth, heads, dropout)
        self.textBlock = DisBlock(dim, depth, heads, dropout)
        
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, dim)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.1)

    def forward(self, vision: Tensor, text: Tensor):
        batch_size = vision.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        multiview_v = self.visionBlock(query_tokens, vision)
        multiview_t = self.textBlock(query_tokens, text)
        return multiview_v, multiview_t

# @TODO:MAIN Model
@MODELS.register_module()
class CLIPSimilarity_split(BaseModel):
    def __init__(
        self,
        data_preprocessor: Dict[str, Dict],
        ismultivew,
        adapter: Dict = None,
        visual_encoder = None,
        text_encoder = None,
        multiview_dis = None,
        to_float32: bool = False,
        class_path = None,
        frozen_layers = False,
        task = "retrieval",
        tau = 0.01,
        loss: Dict = dict(type='NormSoftmaxLoss'),
        multiview_loss = dict(type='MultiviewLoss'),
    ) -> None:
        super(CLIPSimilarity_split,
              self).__init__(data_preprocessor=data_preprocessor)
        self.ismultivew = ismultivew
        self.backbone = MODELS.build(visual_encoder)
        self.text_backbone = MODELS.build(text_encoder)
        if self.ismultivew:
            self.multiview_dis = MODELS.build(multiview_dis)
            self.multiview_loss = MODELS.build(multiview_loss)
        self.loss = MODELS.build(loss)
        self.adapter = MODELS.build(adapter) if adapter is not None else None
        self.task = task
        if self.task == "recognition":
            self.cache_text = True
            self.cache_text_features = None
            with open(class_path,'r') as f:
                classes = f.readlines()
                classes = [c.strip() for c in classes]
                self.text = tokenize(classes)[:,:32]
        self.frozen_layers = frozen_layers
        self.tau = tau
        self._freeze_stages()
        #if self.frozen_layers:
        #    self.text_backbone = self.text_backbone.half()
    
    def init_weights(self):
        pass
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video."""
        frames_features = self.backbone(video)
        return frames_features

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text."""
        if self.frozen_layers:
            with torch.no_grad():
                result = self.text_backbone(text)
        else:
            result = self.text_backbone(text)
        return result

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor]) -> Tuple:
        """Extract features."""
        text_inputs = inputs['text'] if 'text' in inputs else None #@TODO:[16,32]，将最大长度限制改为80，现在太小
        video_inputs = inputs['imgs']
        if text_inputs is None:
            text_features = None
        elif self.task=="recognition" and self.cache_text:
            text_inputs = self.text
            text_inputs = text_inputs.to(video_inputs.device)
            if self.cache_text_features is None:
                self.eval()
                with torch.no_grad():
                    text_features = self.encode_text(text_inputs)
                self.cache_text_features = text_features
                #self.cache_text_features.requires_grad = False
            text_features = self.cache_text_features
        else: # jump to
            text_features = self.encode_text(text_inputs)
        video_features = self.encode_video(video_inputs)

        return video_features, text_features

    #@autocast() @TODO: forward main
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward function."""
        if mode == 'tensor':
            return self.extract_feat(inputs)

        elif mode == 'loss':
            losses = dict()
            video_features, text_features = self.extract_feat(inputs) # [16,512] \ [16, 32, 512] + [16, 512]
            if isinstance(text_features, tuple):
                token_features, text_features = text_features # [16, 32, 512] \ [16, 512]
            if isinstance(video_features, tuple):
                video_tokens, video_features = video_features
                
            if isinstance(self.tau,float):
                logit_scale = 1 / self.tau
            else:
                logit_scale = self.tau.exp()
            
            if self.ismultivew:
            # multiview_embedding cal:
                multiview_v, multiview_t = self.multiview_dis(video_features, text_features) # [batch, num_queries, 512]
                multiview_loss = self.multiview_loss(multiview_v, multiview_t, logit_scale, self.parameters())
                losses['multiview_loss'] = multiview_loss
            else:
                if self.task=='retrieval':
                    logit = None
                    if self.adapter is not None:
                        mask = torch.where(inputs['text']==0,0,1)
                        logit = self.adapter(token_features, video_features, mask)
                    else:
                        video_features = torch.cat(
                            GatherLayer.apply(video_features), dim=0)
                        text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

                    sim_loss = self.loss(video_features, text_features, sim_mat=logit, scale = logit_scale)
                    losses['NCE_loss'] = sim_loss

                elif self.task=='recognition':
                    logits_per_video = logit_scale * sim_matrix(video_features, text_features)
                    labels = [x.gt_labels.item for x in data_samples]
                    labels = torch.stack(labels).to(logits_per_video.device).squeeze()
                    loss = self.loss(logits_per_video,labels)
                    losses['loss'] = loss
                
            return losses

        elif mode == 'predict':
            video_features_t, text_features_t = self.extract_feat(inputs)
            if isinstance(text_features_t, tuple):
                token_features, text_features_t = text_features_t
            if isinstance(video_features_t, tuple):
                video_tokens, video_features_t = video_features_t
            if self.ismultivew:
                video_features, text_features = self.multiview_dis(video_features_t, text_features_t)  # [16, 8, 512]
            else:
                video_features, text_features = video_features_t, text_features_t
            if self.adapter is not None:
                token_id = inputs['text'][0 ] if self.task=="recognition" else inputs['text']
                masks = torch.where(token_id==0,0,1)
                for ds, vf, tf, mask in zip(data_samples, video_features, token_features, masks):
                    tf = token_features if self.task=="recognition" else tf
                    mask = masks if self.task=="recognition" else mask
                    features = BaseDataElement(video_feature=vf, text_feature=tf, mask=mask)
                    ds.features = features
            else:
                for ds, vf, tf in zip(data_samples, video_features, text_features):
                    tf = text_features if self.task=="recognition" else tf
                    features = BaseDataElement(video_feature=vf, text_feature=tf) # [8, 512]
                    ds.features = features
            
            return data_samples

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode') 

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_layers``."""

        if self.frozen_layers:
            for name, param in self.named_parameters():
                if "STAN" in name:
                    continue
                elif 'text_backbone' in name:
                    param.requires_grad = False
                elif 'extra_proj' in name:
                    continue
                elif 'balance' in name:
                    continue
                elif 'backbone.layers' in name:
                    layer_n = int(name.split('.')[2])
                    param.requires_grad = False
                    if layer_n>=20:
                        continue
                    else:
                        continue
                else:
                    param.requires_grad = False
