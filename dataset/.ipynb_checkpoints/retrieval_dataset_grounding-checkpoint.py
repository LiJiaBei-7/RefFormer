import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import numpy as np  

import json, re, random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

import torch

import torchvision


splits_eval = ['test']
splits_train = ['train']
dataset = 'refcocog'

def collate_fn(batch):
    captions = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    targets = torch.cat([item[2] for item in batch], dim=0)
    idx = torch.tensor([int(item[3]) for item in batch], dtype=torch.int32)
    ref_id = torch.tensor([int(item[4]) for item in batch], dtype=torch.int32)
    return captions, images, targets, idx, ref_id

def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id


def ConvertCocoPolysToMask(image, target):
    w, h = image.size

    boxes = torch.tensor(target['bbox']).reshape(-1, 4).detach()
    # guard against no boxes via resizing
    # transform to xyxy
    boxes[:, 2:] += boxes[:, :2]
    # Range crop the coordinates of the bounding box to ensure that the value of the bounding box coordinates
    #  does not exceed the width and height of the image
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    # classes = classes[keep]
    # if self.return_masks:
    #     masks = masks[keep]
    # if keypoints is not None:
    #     keypoints = keypoints[keep]
    target = {}
    target["boxes"] = boxes
    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return image, target


class gd_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.dataset = 'refcocog'
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words

        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_train

        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        self.data_size = len(self.refs_anno)
        print(' ========== Dataset size:', self.data_size)
        
    
    def load_refs(self, idx):
        # here idx is instance index
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        # ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset == 'referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB') 
        box=np.array([self.refs_anno[idx]['bbox']])

        return image, box, self.refs_anno[idx]['iid']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # query
        ref_iter = self.load_refs(index)
        # image and bbox
        image_iter, gt_box_iter, iid= self.load_img_feats(index)
        target = {}
        target['bbox'] = gt_box_iter
        image_iter, target  = ConvertCocoPolysToMask(image_iter, target)
        image_iter, target = self.transform(image_iter, target)
        
        return ref_iter, image_iter, target['boxes'], iid



class gd_train_dataset_full(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.anns = []

        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_train

        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        for split in splits:
            for ann in stat_refs_list[split]:
                iid = ann['iid']
                bbox = ann['bbox']
                for i, ref in enumerate(ann['refs']):
                    data = {}
                    data['iid'] = iid
                    data['ref'] = ref
                    data['bbox'] = bbox
                    data['tid'] = i
                    self.anns.append(data)
                    # refs.append(ref)

        self.data_size = len(self.anns)
        print(' ========== Dataset size:', self.data_size)
        
    
    def load_refs(self, idx):
        # here idx is instance index
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        # ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset =='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB') 
        box=np.array([self.refs_anno[idx]['bbox']])

        return image, box, self.refs_anno[idx]['iid']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        ann = self.anns[index]
        # query
        ref_iter = ann['ref']
        ref_id = ann['tid']
        # image and box
        img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%ann['iid'])
        _image_iter = Image.open(img_path).convert('RGB') 
        gt_box_iter = ann['bbox']
        iid = ann['iid']
        target = {}
        target['bbox'] = gt_box_iter

        image_iter, target = ConvertCocoPolysToMask(_image_iter, target)
        image_iter, target = self.transform(image_iter, target)
        
        return ref_iter, image_iter, target['boxes'], iid, ref_id





class gd_train_dataset_full_faster(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.anns = []

        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_train
        
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]
        
        for split in splits:
            for ann in stat_refs_list[split]:
                iid = ann['iid']
                bbox = ann['bbox']
                for ref in ann['refs']:
                    data = {}
                    data['iid'] = iid
                    data['ref'] = ref
                    data['bbox'] = bbox
                    self.anns.append(data)

        self.data_size = len(self.anns)
        print(' ========== Dataset size:', self.data_size)
        
    
    def load_refs(self, idx):
        # here idx is instance index
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        # ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset =='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB') 
        box=np.array([self.refs_anno[idx]['bbox']])

        return image, box, self.refs_anno[idx]['iid']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        ann = self.anns[index]
        # query
        ref_iter = ann['ref']
        # image and box
        img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%ann['iid'])
        _image_iter = Image.open(img_path).convert('RGB') 
        gt_box_iter = ann['bbox']
        iid = ann['iid']
        target = {}
        target['bbox'] = gt_box_iter

        image_iter, target  = ConvertCocoPolysToMask(_image_iter, target)
        image_iter, target = self.transform(image_iter, target)

        target = target.copy()
        h, w = image_iter.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        
        return ref_iter, image_iter, target['boxes'], iid



class gd_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words

        # ann_file = '/home/ma-user/work/workspace/code/WRef/RefCLIP/data/anns/refcocog.json'
        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_eval

        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        self.data_size = len(self.refs_anno)
        print(' ========== Dataset size:', self.data_size)
        
    
    def load_refs(self, idx):
        # here idx is instance index
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        # ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset =='referit':
            img_path = os.path.join(self.image_root, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB') 
        box=np.array([self.refs_anno[idx]['bbox']])

        return image, box, self.refs_anno[idx]['iid']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # query
        ref_iter = self.load_refs(index)
        # image and bbox
        image_iter, gt_box_iter, iid= self.load_img_feats(index)
        target = {}
        target['bbox'] = gt_box_iter
        image_iter, target  = ConvertCocoPolysToMask(image_iter, target)
        image_iter, target = self.transform(image_iter, target)
        
        return ref_iter, image_iter, target['boxes'], iid





class gd_eval_dataset_faster(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words

        # ann_file = '/home/ma-user/work/workspace/code/WRef/RefCLIP/data/anns/refcocog.json'
        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_eval

        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        self.data_size = len(self.refs_anno)
        print(' ========== Dataset size:', self.data_size)
        
    
    def load_refs(self, idx):
        # here idx is instance index
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        # ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset =='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB') 
        box=np.array([self.refs_anno[idx]['bbox']])

        return image, box, self.refs_anno[idx]['iid']

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # query
        ref_iter = self.load_refs(index)
        # image and bbox
        image_iter, gt_box_iter, iid= self.load_img_feats(index)
        w,h = image_iter.size

        target = {}
        target['bbox'] = gt_box_iter
        image_iter, target  = ConvertCocoPolysToMask(image_iter, target)
        image_iter, _ = self.transform(image_iter, None)

        # h, w = image_iter.shape[-2:]
        # box Normalize
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            # target["boxes"] = boxes
        
        return ref_iter, image_iter, boxes, iid


