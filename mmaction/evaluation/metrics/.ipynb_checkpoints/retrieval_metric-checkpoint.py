# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torch
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from mmaction.models.heads.mug_head import Mug_head
from mmaction.evaluation import top_k_accuracy

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
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

#@TODO:检索指标 测试时使用
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
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)
    
    def compute_metrics(self, results: List) -> Dict:
        if len(results[0]["text_feature"].shape) == 2:
            metrics = self.compute_metrics_multiview(results)
        else:
            metrics = self.compute_metrics_ori(results)
        return metrics
        
    def compute_metrics_ori(self, results: List) -> Dict:
        #TODO:main metrics
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        sim_matrix = text_features @ video_features.T
        # DSL
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

        text_features_flat = text_features.reshape(-1, dim)
        video_features_flat = video_features.reshape(-1, dim) # [8000, 512]

        # Compute pairwise cosine similarity between all text and video features
        similarity_matrix = torch.mm(text_features_flat, video_features_flat.t())

        # Reshape the similarity matrix back to [batch_size, seq_len, batch_size, seq_len]
        similarity_matrix = similarity_matrix.view(batch_size, seq_len, batch_size, seq_len)

        # Average over all seq_len to get the final similarity scores
        mean_similarity = torch.mean(similarity_matrix, dim=1)  # average across text seq_len
        mean_similarity = torch.mean(mean_similarity, dim=2)    # average across video seq_len  [batch, batch]

        # Convert tensor to numpy for ranking calculations
        similarity = mean_similarity.numpy()
        # DSL
        print("with DSL")
        similarity = similarity * np_softmax(similarity*100, axis=0) # DSL
        # ----------------------------------------------------------------
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
        # # Sort and rank the similarities
        # sx = np.argsort(-similarity, axis=1)  # descending sort along each row
        # ranks = np.array([np.where(sx[i] == i)[0][0] for i in range(batch_size)])  # find the rank of the diagonal elements

        # metrics = OrderedDict()
        # for metric in self.metric_list:
        #     if metric == 'R1':
        #         metrics['R1'] = float(np.sum(ranks == 0)) * 100 / batch_size
        #     elif metric == 'R5':
        #         metrics['R5'] = float(np.sum(ranks < 5)) * 100 / batch_size
        #     elif metric == 'R10':
        #         metrics['R10'] = float(np.sum(ranks < 10)) * 100 / batch_size
        #     elif metric == 'MdR':
        #         metrics['MdR'] = np.median(ranks) + 1
        #     elif metric == 'MnR':
        #         metrics['MnR'] = np.mean(ranks) + 1

        # return metrics

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
                 prefix: Optional[str] = None) -> None:
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
            self.results.append(results)

    
    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        mask = np.stack([res['mask'] for res in results])

        similarity = self.Mug_head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask))
        similarity = similarity.numpy()
        print("with DSL")
        similarity = similarity * np_softmax(similarity*100, axis=0) # DSL

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
