import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmaction.registry import MODELS
from einops import rearrange

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cos_norm(a, eps=1e-8):
    if a is None:
        return a
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

@MODELS.register_module()
class NormSoftmaxLoss(nn.Module):
    def forward(self, video_embd=None, text_embd=None, sim_mat=None, scale=None):
        if sim_mat is None:
            # video_embd = video_embd[::, 0, ::]           
            x = sim_matrix(video_embd, text_embd) # [16,16]
            x = x if scale is None else x * scale
        else:
            x = sim_mat if scale is None else sim_mat * scale
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

@MODELS.register_module()
class DualSoftmaxLoss(nn.Module):
    def forward(self, video_embd=None, text_embd=None, sim_mat=None, scale=None):
        if sim_mat is None:           
            x = sim_matrix(video_embd, text_embd) # [16,16]
            x = x if scale is None else x * scale
        else:
            x = sim_mat if scale is None else sim_mat * scale
        
        x = x * F.softmax(x/1000, dim=0)*len(x) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(x, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss
    
class ContrastiveLoss1(nn.Module):
    def __init__(self):
        super(ContrastiveLoss1, self).__init__()
    
    def forward(self, multiview_v, multiview_t):
        assert multiview_v.size() == multiview_t.size()
        cos_similarity = F.cosine_similarity(multiview_v, multiview_t, dim=-1) # [batch_size, num_queries]
        loss = 1 - cos_similarity
        mean_loss = torch.mean(loss)
        
        return mean_loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def compute_loss(self, video_embd=None, text_embd=None, sim_mat=None, scale=None):
        if sim_mat is None:           
            x = sim_matrix(video_embd, text_embd) # [16,16]
            x = x if scale is None else x * scale
        else:
            x = sim_mat if scale is None else sim_mat * scale
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

    def forward(self, video_embd=None, text_embd=None, sim_mat=None, scale=None):
        # multiview_v.shape = [batch_size, seq_len, 512], multiview_t.shape = [batch_size, seq_len, 512]
        batch_size, seq_len, _ = multiview_v.shape
        
        multiview_v = multiview_v.reshape(batch_size * seq_len, -1)
        multiview_t = multiview_t.reshape(batch_size * seq_len, -1)
        return self.compute_loss(multiview_v, multiview_t, sim_mat=sim_mat, scale=scale)

class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()

    def forward(self, multiview_v):
        # multiview_v.shape = [batch_size, seq_len, dim]
        batch_size, seq_len, dim = multiview_v.shape

        norm_v = F.normalize(multiview_v, p=2, dim=2)

        similarity = torch.bmm(norm_v, norm_v.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        # Exclude self-similarity - filling diagonal with 0
        eye_mask = torch.eye(seq_len, device=multiview_v.device).bool()
        similarity.masked_fill_(eye_mask.unsqueeze(0), 0)

        diversity_loss = similarity.sum() / (batch_size * seq_len * (seq_len - 1))
        return diversity_loss


@MODELS.register_module()
class PartialAlignment(nn.Module):
    def __init__(self):
        super(PartialAlignment, self).__init__()
        self.tau=1
        self.margin=0.2
    
    def forward(self, video_embedding, text_embedding, scale, query_token):
        video_embedding = video_embedding[::, -query_token.shape[1]::, ::]  # 只取multiview的部分
        text_embedding = text_embedding[::, -query_token.shape[1]::, ::]

        batch_size, seq_len, dim = video_embedding.size()

        similarity_matrix = torch.einsum('bsd,bsd->bs', video_embedding, text_embedding) / self.tau

        positive_similarity = torch.diag(similarity_matrix)  # 对应位置的相似度
        positive_loss = -torch.log(torch.sigmoid(positive_similarity))

        negative_similarity = similarity_matrix - torch.diag_embed(similarity_matrix)  # 非对应位置的相似度
        negative_loss = torch.clamp(self.margin - torch.log(torch.sigmoid(negative_similarity)), min=0)

        total_loss = positive_loss.sum() + negative_loss.sum()
        return total_loss / (batch_size * seq_len)

@MODELS.register_module()
class MultiviewLoss(nn.Module):
    def __init__(self, weights={}):
        super(MultiviewLoss, self).__init__()
        self.diversity_loss = DiversityLoss()

    def forward(self, video_embd, text_embd):

        dev_l = 0.5 * (self.diversity_loss(text_embd) + self.diversity_loss(video_embd))

        return dev_l
