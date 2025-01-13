import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import hues
import time

from mmaction.registry import MODELS
from mmaction.models.recognizers.clip_similarity import GatherLayer


def norm(a, eps=1e-6):
    a_n = a.norm(dim=-1, keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

@MODELS.register_module()
class DCM(nn.Module):
    def __init__(
        self,
        input_dim = 512, 
        tau = 100,
        retrieval = True,
        communication = False
    ):
        super().__init__()
        self.tau = tau
        self.retrieval = retrieval
        self.fp16_enabled = False
        self.communication = communication
    
    def forward(self, text_feat, video_feat, text_mask, parse_idx=None, parse_mask=None):
        _, T, D = video_feat.size()
        if len(text_mask.size())>=3: # None
            text_mask = text_mask.reshape((-1, ) + text_mask.shape[2:])
        if self.training and torch.cuda.is_available() and self.retrieval: 
            text_feat = torch.cat(GatherLayer.apply(text_feat), dim=0)
            video_feat = torch.cat(GatherLayer.apply(video_feat), dim=0)
            text_mask = torch.cat(GatherLayer.apply(text_mask), dim=0)
            if parse_mask is not None:
                parse_mask = torch.cat(GatherLayer.apply(parse_mask), dim=0)
        B = video_feat.size(0)
        
        if parse_mask is not None:
            text_feat = torch.einsum('atd,at->atd', [text_feat, parse_mask])

        text_feat = norm(text_feat)
        video_feat = norm(video_feat)

        #####################################################################################################
        text_feat_list = []
        text_mask_list = []

        samples_length = text_feat.shape[0]
        idx = 0
        samples_per_batch = 1000

        while idx+samples_per_batch <= samples_length:
            text_feat_list.append(text_feat[idx:idx+samples_per_batch, :, :])
            text_mask_list.append(text_mask[idx:idx+samples_per_batch, :])
            idx += samples_per_batch

        if idx+samples_per_batch > samples_length:
            text_feat_list.append(text_feat[idx::, :, :])
            text_mask_list.append(text_mask[idx::, :])
        #####################################################################################################
        retrieve_logits_list = []
        for i in range(len(text_feat_list)):
            text_feat = text_feat_list[i]
            text_mask = text_mask_list[i]
            # a b: batch size // t v: seq len // d: dim
            retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat]) # 4917 41 512 / 4917 21 512  ---> [1000, 41, 512] [1000, 21, 512]
            
            retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask]) # [8, 8, 32, 12]  

            fill = torch.HalfTensor([float("-inf")]).type_as(retrieve_logits)

            video_pre_softmax1 = torch.softmax(retrieve_logits*self.tau, dim=-1) 
            text_feat1 = torch.einsum('abtv,bvd->abtd', [video_pre_softmax1, video_feat]) # [8, 8, 32, 512]
            ################
            if self.communication:
                Q_t = text_feat1 # [32, 512]
                K_t = text_feat1 # []
                V_t = text_feat1 #
                d_t = Q_t.size(-1)
                attention_scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / (d_t ** 0.5) # 应用softmax得到注意力权重 
                attention_weights = F.softmax(attention_scores, dim=-1)
                text_feat1 = torch.matmul(attention_weights, V_t)
            ##########################
            t2v_logits = torch.einsum('abtd,atd->abt', [text_feat1, text_feat])  # [8, 8, 32]
            
            text_pre_softmax1 = torch.softmax(torch.where(retrieve_logits==0, fill, retrieve_logits)*self.tau, dim=-2)  
            video_feat1 = torch.einsum('abtv,atd->abvd', [text_pre_softmax1, text_feat])
            #############
            if self.communication:
                Q_v = video_feat1 # [32, 512]
                K_v = video_feat1 # []
                V_v = video_feat1 #
                d_v = Q_v.size(-1)
                attention_scores = torch.matmul(Q_v, K_v.transpose(-2, -1)) / (d_v ** 0.5) # 应用softmax得到注意力权重 
                attention_weights = F.softmax(attention_scores, dim=-1)
                video_feat1 = torch.matmul(attention_weights, V_v)
            ###############
            v2t_logits = torch.einsum('abvd,bvd->abv', [video_feat1, video_feat])
            
            text_pre_softmax2 = torch.softmax(torch.where(t2v_logits==0, fill, t2v_logits)*self.tau, dim=-1)  
            video_feat2 = torch.einsum('abt,atd->abd', [text_pre_softmax2, text_feat])

            video_pre_softmax2 = torch.softmax(v2t_logits*self.tau, dim=-1) 
            text_feat2 = torch.einsum('abv,bvd->abd', [video_pre_softmax2, video_feat])
        
            retrieve_logits = torch.einsum('abd,abd->ab', [video_feat2, text_feat2]) # [均为8， 8， 512]， 得到[8, 8] sim_martix

            retrieve_logits_list.append(retrieve_logits)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)

        return retrieve_logits
