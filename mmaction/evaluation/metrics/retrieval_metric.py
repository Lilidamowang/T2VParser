# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import hues
import torch.nn.functional as F
import torch
import time
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from mmaction.models.heads.mug_head import Mug_head
from mmaction.evaluation import top_k_accuracy
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_features, gallery_features):
    return cosine_similarity(query_features, gallery_features)

def compute_metrics_des(similarity_matrix, gt_labels, k_values=[1, 5, 10]):
    ranks = np.argsort(-similarity_matrix, axis=1)
    metrics = {}
    for k in k_values:
        recall_at_k = (np.sum(np.any(ranks[:, :k] == gt_labels[:, None], axis=1)) / len(gt_labels))
        metrics[f'R{k}'] = recall_at_k
    
    mean_rank = np.mean(np.argmax(ranks == gt_labels[:, None], axis=1) + 1)
    median_rank = np.median(np.argmax(ranks == gt_labels[:, None], axis=1) + 1)
    metrics['MdR'] = median_rank
    metrics['MnR'] = mean_rank
    
    return metrics

def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p

@METRICS.register_module()
class RetrievalMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 twostage,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list
        self.twostage = twostage

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            # mask = features['mask'].cpu().numpy()
            results['video_feature'] = video_feature
            # results['mask'] = mask
            results['text_feature'] = text_feature
            self.results.append(results)
    
    def compute_metrics(self, results: List) -> Dict:
        if len(results[0]["text_feature"].shape) == 2:
            if self.twostage:
                metrics = self.compute_metrics_twostage(results)
            else:
                metrics = self.compute_metrics_multiview(results)
        else:
            metrics = self.compute_metrics_ori(results)
        return metrics
    
    def compute_metrics_temp(self, video_features, text_features):

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        sim_matrix = text_features @ video_features.T
        print("with DSL")
        sim_matrix = sim_matrix * np_softmax(sim_matrix*100, axis=0)
        sx = np.sort(-sim_matrix)
        d = np.diag(-sim_matrix)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1
        return metrics

    def compute_metrics_ori(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        sim_matrix = text_features @ video_features.T
        print("with DSL")
        sim_matrix = sim_matrix * np_softmax(sim_matrix*100, axis=0) # DSL
        # ----------------------------------------------------------------
        sx = np.sort(-sim_matrix)
        d = np.diag(-sim_matrix)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1
        return metrics
    
    def compute_metrics_multiview(self, results: List): 
        text_features = []
        video_features = []
        for item in results: # target: [batch, 8, 512]
            text_features.append(item['text_feature'])  # [8, 512]
            video_features.append(item['video_feature'])
        text_features = torch.tensor(text_features)  # [1000, 8, 512]
        video_features = torch.tensor(video_features)

        batch_size, seq_len, dim = text_features.shape

        text_features = text_features.reshape(-1, dim)
        video_features = video_features.reshape(-1, dim) # [8000, 512]

        # Compute pairwise cosine similarity between all text and video features
        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        sim_matrix = text_features @ video_features.T
        sim_matrix = sim_matrix.reshape((batch_size, seq_len, batch_size, seq_len))
        sim_matrix = sim_matrix.mean(axis=(1, 3))

        sim_matrix = sim_matrix * np_softmax(sim_matrix*100, axis=0) # DSL
        sx = np.sort(-sim_matrix)
        d = np.diag(-sim_matrix)
        ind = np.where((sx - d[:, None]) == 0)[1]
        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1
        return metrics

    def compute_metrics_twostage(self, results: List, k=50): 
        text_features = []
        video_features = []
        for item in results:
            text_features.append(item['text_feature']) 
            video_features.append(item['video_feature'])
        text_features = torch.tensor(text_features) 
        video_features = torch.tensor(video_features)
        
        batch_size, seq_len, dim = text_features.shape

        text_first_features = text_features[:, 0, :] 
        video_first_features = video_features[:, 0, :] 

        text_first_features = text_first_features / np.linalg.norm(text_first_features, axis=-1, keepdims=True)
        video_first_features = video_first_features / np.linalg.norm(video_first_features, axis=-1, keepdims=True)

        sim_matrix_first = text_first_features @ video_first_features.T 
        top_k_indices = torch.topk(sim_matrix_first, k=k, dim=1).indices

        text_remaining_features = text_features[:, 1:, :].reshape(-1, dim)  
        video_remaining_features = video_features[:, 1:, :].reshape(-1, dim) 

        text_remaining_features = text_remaining_features / np.linalg.norm(text_remaining_features, axis=-1, keepdims=True)
        video_remaining_features = video_remaining_features / np.linalg.norm(video_remaining_features, axis=-1, keepdims=True)

        sim_matrix_second = text_remaining_features @ video_remaining_features.T 
        sim_matrix_second = sim_matrix_second.reshape((batch_size, seq_len-1, batch_size, seq_len-1))
        sim_matrix_second = sim_matrix_second.mean(axis=(1, 3))
       
        final_sim_matrix = np.full((batch_size, batch_size), -0.99)
        for i in range(batch_size): 
            top_k = top_k_indices[i] 
            for k_idx in top_k:
                final_sim_matrix[i, k_idx] = sim_matrix_second[i, k_idx]
            

        final_sim_matrix = final_sim_matrix * np_softmax(final_sim_matrix * 100, axis=0) 
        sx = np.sort(-final_sim_matrix)
        d = np.diag(-final_sim_matrix)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
        metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
        metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
        metrics['MdR'] = np.median(ind) + 1
        metrics['MnR'] = np.mean(ind) + 1

        return metrics

    def compute_metrics_twostage_2(self, results: List, k=50): 
        text_features = []
        video_features = []
        for item in results:
            text_features.append(item['text_feature'])
            video_features.append(item['video_feature'])
        text_features = torch.tensor(text_features)
        video_features = torch.tensor(video_features)

        
        batch_size, seq_len, dim = text_features.shape

    
        text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        video_features = video_features / torch.norm(video_features, dim=-1, keepdim=True)

        first_text_feature = text_features[:, 0, :]  # Shape: [1000, 512]
        first_video_feature = video_features[:, 0, :]  # Shape: [1000, 512]

        first_sim_matrix = first_text_feature @ first_video_feature.T  # Shape: [1000, 1000]

        top_k_values, top_k_indices = torch.topk(first_sim_matrix, k=k, dim=1)  # Shape: [1000, k]

        remaining_text_features = text_features[:, 1:, :]  # Shape: [1000, 8, 512]
        remaining_video_features = video_features[:, 1:, :]  # Shape: [1000, 8, 512]

        ranks = []

        for i in range(batch_size):
            candidate_video_features = remaining_video_features[top_k_indices[i]] 

            sim_matrix = remaining_text_features[i].unsqueeze(0) @ candidate_video_features.permute(0, 2, 1)
            sim_matrix = sim_matrix.mean(axis=(1, 3)).squeeze(0) 

            sorted_indices = torch.argsort(-sim_matrix) 
            correct_index = torch.nonzero(top_k_indices[i] == i, as_tuple=False).item()
            correct_rank = torch.where(sorted_indices == correct_index)[0].item()

            ranks.append(correct_rank + 1) 


        metrics = OrderedDict()
        ranks = np.array(ranks)

        metrics['R1'] = np.mean(ranks <= 1) * 100
        metrics['R5'] = np.mean(ranks <= 5) * 100
        metrics['R10'] = np.mean(ranks <= 10) * 100
        metrics['MdR'] = np.median(ranks)
        metrics['MnR'] = np.mean(ranks)

        return metrics


@METRICS.register_module()
class PostProc_RetrievalMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None,
                 DSL=True) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")
        self.Mug_head = Mug_head()
        self.Mug_head.eval()
        self.metric_list = metric_list
        self.DSL = DSL

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            mask = features['mask'].cpu().numpy()
            results['video_feature'] = video_feature
            results['mask'] = mask
            results['text_feature'] = text_feature
            # results['mv'] = features['mv'].cpu().numpy()
            # results['mt'] = features['mt'].cpu().numpy()
            self.results.append(results)

    
    def compute_metrics(self, results: List) -> Dict:

        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        mask = np.stack([res['mask'] for res in results])

        similarity = self.Mug_head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask))
        similarity = similarity.detach().numpy()
        
        if self.DSL:
            hues.info("with DSL")
            similarity = similarity * np_softmax(similarity*100, axis=0)

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1
        return metrics

@METRICS.register_module()
class PostProc_RetrievalMetric_Save(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None,
                 DSL=True) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")
        self.Mug_head = Mug_head()
        self.Mug_head.eval()
        self.metric_list = metric_list
        self.DSL = DSL

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            mask = features['mask'].cpu().numpy()
            results['video_feature'] = video_feature
            results['mask'] = mask
            results['text_feature'] = text_feature
            results['mv'] = features['mv'].cpu().numpy()
            results['mt'] = features['mt'].cpu().numpy()
            self.results.append(results)

    
    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        mv = np.stack([res['mv'] for res in results])
        mt = np.stack([res['mt'] for res in results])
        mask = np.stack([res['mask'] for res in results])

        np.save('temp_out_data/emb_v_msvd2.npy', mv)
        np.save('temp_out_data/emb_t_msvd2.npy', mt)

        similarity = self.Mug_head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask))
        similarity = similarity.detach().numpy()
        
        if self.DSL:
            hues.info("with DSL")
            # similarity = similarity * np_softmax(similarity*100, axis=0) # DSL
            similarity = torch.from_numpy(similarity)
            similarity = similarity * F.softmax(similarity/1, dim=0)*len(similarity)
            similarity = similarity.numpy()

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1

        return metrics

@METRICS.register_module()
class ZeroShotAccMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            label = data_sample['gt_labels']
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            result['video_feature'] = video_feature
            if not hasattr(self,"text_feature"):
                self.text_feature = text_feature
            
            if 'mask' in features:
                if not hasattr(self,"mask"):
                    self.mask = features['mask'].cpu().numpy()
            
            if label['item'].size(0) == 1:
                # single-label
                result['label'] = label['item'].item()
            else:
                # multi-label
                result['label'] = label['item'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = self.text_feature
        labels = [x['label'] for x in results]

        if hasattr(self, 'mask'):
            mask = self.mask
            head = Mug_head()
            score = head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask)).numpy()
            score = score.T

        else:
            video_features = video_features / np.linalg.norm(
                video_features, axis=-1, keepdims=True)
            text_features = text_features / np.linalg.norm(
                text_features, axis=-1, keepdims=True)
            score = video_features @ text_features.T

        top_k_acc = top_k_accuracy(score, labels, (1,5))
        metrics = {}
        metrics['overall_acc1'] = top_k_acc[0]
        metrics['overall_acc5'] = top_k_acc[1]

        return metrics
