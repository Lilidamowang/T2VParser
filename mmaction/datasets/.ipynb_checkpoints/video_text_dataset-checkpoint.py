# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import csv
import os.path as osp
from typing import Callable, Dict, List, Optional, Union
from collections import OrderedDict

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset
from .transforms.text_transforms import tokenize
from .video_dataset import VideoDataset

#@TODO:MSRVTT dataset
@DATASETS.register_module()
class MsrvttDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        if self.ann_file.endswith("train_9k.json"):
            all_train_id = []
            with open("/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/MSRVTT_train.9k.csv", 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    all_train_id.append(row[0])
            with open("/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/MSRVTT_data.json", 'r') as f:
                res = json.load(f)
                sentences = res['sentences']
                for item in sentences:
                    cap = item['caption']
                    vid = item['video_id']
                    if vid in all_train_id:
                        vpath = f"/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/msrvttvideo/{vid}.mp4"
                        data_item = dict(filename=vpath, text=cap)
                        video_text_pairs=[]
                        video_text_pairs.append(data_item)
                        data_list.extend(video_text_pairs)
            with open('/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/msrvtt_doc.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] in all_train_id:
                        vpath = f"/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/msrvttvideo/{row[0]}.mp4"
                        data_item = dict(filename=vpath, text=row[1])
                        video_text_pairs=[]
                        video_text_pairs.append(data_item)
                        data_list.extend(video_text_pairs)
            print(f"all training dataset size: {len(data_list)}")
        else:
            with open("/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/MSRVTT_JSFUSION_test.csv", 'r') as f:
                reader = csv.reader(f)
                all_row = []
                for row in reader:
                    all_row.append(row)
                del all_row[0]
            for item in all_row:
                vid = item[2]
                cap = item[3]
                vpath = f"/root/autodl-fs/root/autodl-tmp/data/msrvtt_data/msrvttvideo/{vid}.mp4"
                data_item = dict(filename=vpath, text=cap)
                video_text_pairs=[]
                video_text_pairs.append(data_item)
                data_list.extend(video_text_pairs)
        return data_list
    
@DATASETS.register_module()
class DidemoDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            video_dict = json.load(f)
            for filename, texts in video_dict.items():
                filename = osp.join(self.data_prefix['video'], filename)
                text = " ".join(texts)
                data_item = dict(filename=filename, text=text)
                data_list.append(data_item)

        return data_list
    
@DATASETS.register_module()
class ActivityNetVideoDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            videos = f.readlines()
            for video in videos:
                video = video.strip()
                video,frame = video.split(',')
                frame_dir = osp.join(self.data_root,video)
                data_item = dict(file_name=video, frame_dir=frame_dir, total_frames=int(frame), filename_tmpl="{:0>6}.jpg", offset=1)
                data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class WeividDataset_filtered(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        with open(self.ann_file) as f:
            video_dict = json.load(f)
            for video_dir, text in video_dict.items():
                filename = os.path.join(self.data_root,video_dir)
                info = {'filename':filename, 'text':text}
                data_list.append(info)
        return data_list
    
@DATASETS.register_module()
class LsmdcDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        with open(self.ann_file) as f:
            video_dict = json.load(f)
            for video_dir, text in video_dict.items():
                info = {'filename':video_dir, 'text':text}
                data_list.append(info)
        return data_list
    
@DATASETS.register_module()
class ZeroShotClfDataset(VideoDataset):
    def __init__(self, class_path, label_offset=0, **kwargs):
        self.label_offset = label_offset
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, label = line_split[0], line_split[1:]
                label = list(map(int, label))
            else:
                filename, label = line_split
                label = int(label) + self.label_offset
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label, text=[0]))
        return data_list

@DATASETS.register_module()
class CC_Dataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        reader = csv.reader(open(self.ann_file))
        for i,line in enumerate(reader):
            content = line[0]
            j = 1
            while '\t' not in content:
                content += line[j]
                j += 1
            caption, video_dir = content.split('\t')
            filename = os.path.join(self.data_root,video_dir)
            
            info = {'img_path':filename, 'text':caption}
            data_list.append(info)
        return data_list
    
@DATASETS.register_module()
class CoCo_Dataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        anno_info = json.load(open(self.ann_file, 'r'))

        img_dict = OrderedDict()
        for idx, img in enumerate(anno_info['images']):
            if img['id'] not in img_dict:
                img_rel_path = img['coco_url'].rsplit('/', 2)[-2:]
                img_path = os.path.join(self.data_root, *img_rel_path)

                # create new idx for image
                img_dict[img['id']] = dict(
                    img_path=img_path,
                )

        for idx, anno in enumerate(anno_info['annotations']):
            anno['text'] = anno.pop('caption')
            train_data = anno.copy()
            train_image = img_dict[train_data['image_id']]
            train_data['img_path'] = train_image['img_path']
            data_list.append(train_data)
        return data_list

@DATASETS.register_module()
class VG_Dataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        anno_info = json.load(open(self.ann_file, 'r'))
        for idx, anno in enumerate(anno_info):
            img_path = anno['image'].split('/')[-1]
            VG_path = 'VG_100K' if anno['dir_id']==1 else 'VG_100K_2'
            img_path = os.path.join(self.data_root,VG_path,img_path)
            info = {'img_path':img_path, 'text':anno['caption']}
            data_list.append(info)
        return data_list    
        