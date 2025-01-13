# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import time

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from Bert import BertConfig, BertLMHeadModel
from MultiviewDecoder import MultiviewLayer, MultiviewDecoder

from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel
from mmengine.structures import InstanceData, BaseDataElement
from mmengine.runner import autocast

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, OptSampleList
from mmaction.models.losses import sim_matrix
from mmaction.datasets.transforms.text_transforms import tokenize
import torch.nn.functional as F


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
    def __init__(self, dim, dropout=0.1):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(dim, dim),
            # nn.Dropout(dropout)
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

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


        for _ in range(depth):

            self.self_attn_layers.append(
                nn.MultiheadAttention(dim, heads, batch_first=True)
            )

            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    dim,
                    heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            self.ffn_layers.append(
                FFN(dim, dropout)
            )

    def forward(self, MultiViewQuery: Tensor, info: Tensor) -> Tensor:
        for self_attn, cross_attn, ffn in zip(
            self.self_attn_layers,
            self.cross_attn_layers,
            self.ffn_layers,
        ):
            residual = MultiViewQuery
            MultiViewQuery, _ = self_attn(MultiViewQuery, MultiViewQuery, MultiViewQuery) # 自注意力
            MultiViewQuery = residual + MultiViewQuery

            residual = MultiViewQuery
            MultiViewQuery, _ = cross_attn(MultiViewQuery, info, info) # 交叉注意力
            MultiViewQuery = residual + MultiViewQuery

            residual = MultiViewQuery
            MultiViewQuery = ffn(MultiViewQuery) # 前馈网络
            MultiViewQuery = residual + MultiViewQuery

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
        print(f"depth : {depth} \n heads : {heads} \n num_query : {num_query_token}")
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

class DisBlock_Bert(nn.Module):

    def __init__(
        self,
        dim: int,
        num_query_token:int = 8,
        dropout: float = 0.1,
        bert_weight = "../weight/Bert",
        *args,
        **kwargs,
    ):
        super(DisBlock_Bert, self).__init__(*args, **kwargs) 
        encoder_config = BertConfig.from_pretrained(bert_weight)
        encoder_config.encoder_width = dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.is_decoder = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        self.encoder_config = encoder_config

        self.Block = BertLMHeadModel.from_pretrained(
            bert_weight, config=encoder_config
        )
        self.Block.resize_token_embeddings(dim)
        state_dict = self.Block.state_dict()
        for name, param in self.Block.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, MultiViewQuery, info):
        info_atts = torch.ones(info.size()[:-1], dtype=torch.long).to(
            info.device
        )
        query_output = self.Block.bert(
            query_embeds=MultiViewQuery,
            encoder_hidden_states=info,
            encoder_attention_mask=info_atts,
            use_cache=True,
            return_dict=True,
        )
        # print(query_output.last_hidden_state.shape)
        return query_output.last_hidden_state

@MODELS.register_module()
class MultiViewDis_Bert(nn.Module):

    def __init__(
        self,
        dim: int,
        num_query_token:int = 8,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        print("===MultiViewDis_Bert Init===")
        super(MultiViewDis_Bert, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_query_token = num_query_token
        self.dropout = dropout

        self.dis_block_T = DisBlock_Bert(dim, num_query_token)
        self.dis_block_V = DisBlock_Bert(dim, num_query_token) 

        self.linear_proj_v = nn.Sequential(
            nn.Linear(self.dis_block_V.encoder_config.hidden_size, self.dis_block_V.encoder_config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dis_block_V.encoder_config.hidden_size * 4, dim),
            nn.Dropout(dropout)
        )

        self.linear_proj_t = nn.Sequential(
            nn.Linear(self.dis_block_T.encoder_config.hidden_size, self.dis_block_T.encoder_config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dis_block_T.encoder_config.hidden_size * 4, dim),
            nn.Dropout(dropout)
        )

        # self.proj_v = nn.Linear(dim, self.dis_block_V.encoder_config.hidden_size)
        # self.proj_t = nn.Linear(dim, self.dis_block_T.encoder_config.hidden_size)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.dis_block_V.encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=self.dis_block_V.encoder_config.initializer_range)
        
        self.query_tokens = query_tokens

    def forward(self, vision, text):
        batch_size = vision.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        # vision = self.proj_v(vision)
        # text = self.proj_t(text)
        multiview_v = self.dis_block_V(query_tokens, vision)
        multiview_t = self.dis_block_T(query_tokens, text)
        multiview_v = self.linear_proj_v(multiview_v)
        multiview_t = self.linear_proj_t(multiview_t)
        return multiview_v, multiview_t

@MODELS.register_module()
class MultiviewDis_Decoder(nn.Module):
    def __init__(self, d_model, num_query_token, nhead, num_layers, dropout=0.1):
        super(MultiviewDis_Decoder, self).__init__()
        # self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self_attention=None
        multiview_layer = lambda: MultiviewLayer(d_model, nhead, self_attention=self_attention)
        
        self.dis_block_T = MultiviewDecoder(multiview_layer, num_layers)
        self.dis_block_V = MultiviewDecoder(multiview_layer, num_layers)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, d_model)
        )
        query_tokens.data.normal_(mean=0.0, std=0.1)
        
        self.query_tokens = query_tokens
        print("===MultiviewDis_Decoder Init===")

    def forward(self, vision, text):
        batch_size = vision.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        multiview_v = self.dis_block_V(query_tokens, vision)
        multiview_t = self.dis_block_T(query_tokens, text)
        return multiview_v, multiview_t
        
@MODELS.register_module()
class CLIPSimilarity_split(BaseModel):
    def __init__(
        self,
        data_preprocessor: Dict[str, Dict],
        ismultivew = False,
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
        concat = False,
        plus = False,
        frozen=False,
        concat_local = False,
    ) -> None:
        super(CLIPSimilarity_split,
              self).__init__(data_preprocessor=data_preprocessor)
        self.ismultivew = ismultivew
        self.backbone = MODELS.build(visual_encoder)
        self.text_backbone = MODELS.build(text_encoder)
        if self.ismultivew:
            self.multiview_dis = MODELS.build(multiview_dis)
            self.multiview_loss = MODELS.build(multiview_loss)
            self.video_mlp = FFN(self.multiview_dis.dim)
        self.loss = MODELS.build(loss)
        self.adapter = MODELS.build(adapter) if adapter is not None else None
        self.task = task
        self.concat = concat
        self.plus = plus
        self.concat_local = concat_local
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
        text_inputs = inputs['text'] if 'text' in inputs else None
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

            video_features, text_features = self.extract_feat(inputs) 
            if isinstance(text_features, tuple):
                token_features, text_features = text_features 
            if isinstance(video_features, tuple):
                video_tokens, video_features = video_features
            
            if isinstance(self.tau,float):
                logit_scale = 1 / self.tau
            else:
                logit_scale = self.tau.exp()
            
            if self.ismultivew:
                
                video_features_MLP = self.video_mlp(video_features)
                multiview_v, multiview_t = self.multiview_dis(video_features_MLP, token_features) # [batch, num_queries, 512]
               
                video_features_local = video_features
                video_features = video_features.mean(1)

                if self.plus:
                    normalized_multiview_v = F.normalize(multiview_v, p=2, dim=-1)
                    normalized_video_features = F.normalize(video_features, p=2, dim=-1)
                    expanded_video_features = normalized_video_features.unsqueeze(1).expand_as(normalized_multiview_v)
                    multiview_v = normalized_multiview_v + expanded_video_features

                    normalized_multiview_t = F.normalize(multiview_t, p=2, dim=-1)
                    normalized_text_features = F.normalize(text_features, p=2, dim=-1)
                    expanded_text_features = normalized_text_features.unsqueeze(1).expand_as(normalized_multiview_t)
                    multiview_t = normalized_multiview_t + expanded_text_features
                if self.concat_local:
                    multiview_v = torch.cat((video_features_local, multiview_v), dim=1).contiguous()
                    multiview_t = torch.cat((token_features, multiview_t), dim=1).contiguous()
                if self.concat:
                    multiview_v = torch.cat((video_features.unsqueeze(1), multiview_v), dim=1).contiguous()
                    multiview_t = torch.cat((text_features.unsqueeze(1), multiview_t), dim=1).contiguous()
            
                video_features = multiview_v
                text_features = multiview_t
                logit = None
                if self.adapter is not None:
                    mask_glo = torch.ones(text_features.shape[0], 1, dtype=torch.int32).to(text_features.device)
                    mask_local = torch.where(inputs['text']==0,0,1).to(text_features.device)
                    mask_multiview = torch.ones(text_features.shape[0], self.multiview_dis.query_tokens.shape[1], dtype=torch.int32).to(text_features.device)
                    if self.concat and self.concat_local:
                        mask = torch.concat([mask_glo, mask_local, mask_multiview], dim=1).to(text_features.device)
                    elif self.concat:
                        mask = torch.concat([mask_glo, mask_multiview], dim=1).to(text_features.device)
                    elif self.concat_local:
                        mask = torch.concat([mask_local, mask_multiview], dim=1).to(text_features.device)
        
                    logit = self.adapter(text_features, video_features, mask)
                else:
                    video_features = torch.cat(
                            GatherLayer.apply(video_features), dim=0)
                    text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                if self.adapter is None:
                    b,q,d = video_features.shape
                    video_features = video_features.reshape(b*q, d)
                    text_features = text_features.reshape(b*q, d)
                sim_loss = self.loss(video_features, text_features, sim_mat=logit, scale = logit_scale)
                if self.adapter is None:
                    video_features = video_features.reshape(b, q, d)
                    text_features = text_features.reshape(b, q, d)

                video_embd = video_features[::, -self.multiview_dis.query_tokens.shape[1]::, ::]  # 只取multiview的部分
                text_embd = text_features[::, -self.multiview_dis.query_tokens.shape[1]::, ::]
                multiview_loss =0.08 * self.multiview_loss(video_embd, text_embd)
            
                losses['NCE_loss'] = sim_loss
                losses['Multiview_loss'] = multiview_loss
            

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
                    video_features = video_features.mean(1)
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
            all_embeddings_v = []
            all_embeddings_t = []
            video_features, text_features = self.extract_feat(inputs)
        
            if isinstance(text_features, tuple):
                token_features, text_features = text_features
            if isinstance(video_features, tuple):
                video_tokens, video_features = video_features
            if self.ismultivew:
                
                video_features_MLP = self.video_mlp(video_features)
                multiview_v_before, multiview_t_before = self.multiview_dis(video_features_MLP, token_features) 
                multiview_v = multiview_v_before
                multiview_t = multiview_t_before

                video_features_local = video_features
                video_features = video_features.mean(1) 
                if self.plus:
                    normalized_multiview_v = F.normalize(multiview_v, p=2, dim=-1)
                    normalized_video_features = F.normalize(video_features, p=2, dim=-1)
                    expanded_video_features = normalized_video_features.unsqueeze(1).expand_as(normalized_multiview_v)
                    multiview_v = normalized_multiview_v + expanded_video_features

                    normalized_multiview_t = F.normalize(multiview_t, p=2, dim=-1)
                    normalized_text_features = F.normalize(text_features, p=2, dim=-1)
                    expanded_text_features = normalized_text_features.unsqueeze(1).expand_as(normalized_multiview_t)
                    multiview_t = normalized_multiview_t + expanded_text_features
                if self.concat_local:
                    multiview_v = torch.cat((video_features_local, multiview_v), dim=1).contiguous()
                    multiview_t = torch.cat((token_features, multiview_t), dim=1).contiguous()
                if self.concat:
        
                    multiview_v = torch.cat((video_features.unsqueeze(1), multiview_v), dim=1).contiguous()
                    multiview_t = torch.cat((text_features.unsqueeze(1), multiview_t), dim=1).contiguous()

                token_features = multiview_t
                video_features = multiview_v
    
                for item in multiview_v_before:
                    all_embeddings_v.append(item)
                for item in multiview_t_before:
                    all_embeddings_t.append(item)

            else:
                if self.adapter is not None:
                    mask = torch.where(inputs['text']==0,0,1)
                    logit = self.adapter(token_features, video_features, mask)
                    # video_features, text_features = self.adapter(token_features, video_features, mask)
                else:
                    video_features, text_features = video_features.mean(1), text_features
                    token_features = text_features
                    

            if self.adapter is not None and not self.ismultivew:
                token_id = inputs['text'][0 ] if self.task=="recognition" else inputs['text']
                masks = torch.where(token_id==0,0,1)
                for ds, vf, tf, mask in zip(data_samples, video_features, token_features, masks):
                    tf = token_features if self.task=="recognition" else tf
                    mask = masks if self.task=="recognition" else mask
                    features = BaseDataElement(video_feature=vf, text_feature=tf, mask=mask)
                    ds.features = features
            else:
                token_id = inputs['text'][0 ] if self.task=="recognition" else inputs['text']
                # masks = torch.where(token_id==0,0,1)
                mask_glo = torch.ones(text_features.shape[0], 1, dtype=torch.int32).to(text_features.device)
                mask_local = torch.where(inputs['text']==0,0,1).to(text_features.device)
                mask_multiview = torch.ones(text_features.shape[0], self.multiview_dis.query_tokens.shape[1], dtype=torch.int32).to(text_features.device)
    
                if self.concat and self.concat_local:
                    masks = torch.concat([mask_glo, mask_local, mask_multiview], dim=1).to(text_features.device)
                elif self.concat:
                    masks = torch.concat([mask_glo, mask_multiview], dim=1).to(text_features.device)
                elif self.concat_local:
                    masks = torch.concat([mask_local, mask_multiview], dim=1).to(text_features.device)
                for ds, vf, tf, mask, mv, mt in zip(data_samples, video_features, token_features, masks, all_embeddings_v, all_embeddings_t):
                    tf = token_features if self.task=="recognition" else tf
                    # mask = torch.ones(token_features.shape[1], dtype=torch.int32)
                    mask = masks if self.task=="recognition" else mask
                    features = BaseDataElement(video_feature=vf, text_feature=tf, mask=mask, mv=mv, mt=mt)
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
