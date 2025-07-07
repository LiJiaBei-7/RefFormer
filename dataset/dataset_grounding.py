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


splits_eval = ['val']
splits_train = ['train']
dataset = 'refcocog'

def collate_fn(batch):
    captions = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    targets = torch.cat([item[2] for item in batch], dim=0)
    idx = torch.tensor([int(item[3]) for item in batch], dtype=torch.int32)
    # ref_id = torch.tensor([int(item[4]) for item in batch], dtype=torch.int32)
    ref_id = [item[4] for item in batch]
    return captions, images, targets, idx, ref_id


def referIt_flickr30k_collate_fn(batch):
    captions = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    targets = torch.cat([item[2] for item in batch], dim=0)
    idx = torch.tensor([int(item[3]) for item in batch], dtype=torch.int32)
    ref_id = torch.zeros(len(batch), dtype=torch.int32)
    return captions, images, targets, idx, ref_id


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id


# refcoco 的读取box 坐标为 xywh
def ConvertCocoPolysToMask(image, target, use_keep=True, convert_bbox=True):
    w, h = image.size

    boxes = torch.tensor(target['bbox']).reshape(-1, 4).detach()
    # guard against no boxes via resizing
    # transform to xyxy
    if convert_bbox:
        boxes[:, 2:] += boxes[:, :2]
    # Range crop the coordinates of the bounding box to ensure that the value of the bounding box coordinates
    #  does not exceed the width and height of the image
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    if use_keep:
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
    boxes = boxes.type(torch.float32)
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






class Flickr30k_dataset(Dataset):
    def __init__(self, ann_file_list, transform, image_root, max_words=30, splits_eval='val'):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.anns = []

        # bbox format x1y1x2y2
        stat_refs_list = []
        if isinstance(ann_file_list, list):
            for ann_file in ann_file_list:
                stat_refs_list = torch.load(ann_file)
        else:
            stat_refs_list = torch.load(ann_file_list)
        # ('100652400.jpg', array([ 53.,  45., 110., 203.], dtype=float32), 'man')

        for stat_refs in stat_refs_list:
            data = {}
            data['image_file_name'], data['bbox'], data['ref'] = stat_refs[0], stat_refs[1], stat_refs[2], 
            self.anns.append(data)

        self.data_size = len(self.anns)
        print(' ========== Dataset size:', self.data_size)
        


    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        ann = self.anns[index]
        # query
        ref_iter = ann['ref']
        image_name = ann['image_file_name']
        # image and box
        img_path=os.path.join(self.image_root, image_name)
        _image_iter = Image.open(img_path).convert('RGB') 
        gt_box_iter = ann['bbox']
        target = {}
        target['bbox'] = gt_box_iter

        image_iter, target = ConvertCocoPolysToMask(_image_iter, target, convert_bbox=False)
        image_iter, target = self.transform(image_iter, target)
        
        return ref_iter, image_iter, target['boxes'], index



class gd_train_dataset_full(Dataset):
    def __init__(self, ann_file_list, transform, image_root, max_words=30):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.anns = []

        stat_refs_list = []
        for ann_file in ann_file_list:
            stat_refs_list.append(json.load(open(ann_file, 'r')))

        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        # stat_refs_list=json.load(open(ann_file, 'r'))
        splits = splits_train

        # self.refs_anno=[]
        # for split_ in splits:
        #     self.refs_anno+= stat_refs_list[split_]

        for split in splits:
            for stat_refs in stat_refs_list:
                for ann in stat_refs[split]:
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




class referIt_dataset(Dataset):
    def __init__(self, ann_file_list, transform, image_root, max_words=30, splits_eval='test'):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.anns = []

        if isinstance(ann_file_list, list):
            stat_refs_list = torch.load(ann_file_list[0])
        else:
            stat_refs_list = torch.load(ann_file_list)
        # ('19059.jpg', '19059_10.pth', [0, 255, 28, 310], 'small tree on left on screen', 
        # [('r1', ['tree']), ('r2', ['none']), ('r3', ['small']), ('r4', ['none']), ('r5', ['none']), ('r6', ['none']), ('r7', ['none']), ('r8', ['screen', 'left'])])
 
        splits = splits_train

        for ann in stat_refs_list:
            data = {}
            data['image_name'], data['bbox'], data['ref'] = ann[0], ann[2], ann[3]
            self.anns.append(data)

        self.data_size = len(self.anns)
        print(' ========== Dataset size:', self.data_size)
        
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        ann = self.anns[index]
        # query
        ref_iter = ann['ref']
        image_name = ann['image_name']
        # image and box
        img_path=os.path.join(self.image_root, image_name)
        _image_iter = Image.open(img_path).convert('RGB') 
        gt_box_iter = ann['bbox']
        target = {}
        target['bbox'] = gt_box_iter

        image_iter, target = ConvertCocoPolysToMask(_image_iter, target)
        image_iter, target = self.transform(image_iter, target)
        
        return ref_iter, image_iter, target['boxes'], index






class gd_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, splits_eval='val', mix=True):
        self.dataset = dataset
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words

        # ann_file = '/home/ma-user/work/workspace/code/WRef/RefCLIP/data/anns/refcocog.json'
        # dict dict_keys(['train', 'val', 'testA', 'testB'])
        stat_refs_list=json.load(open(ann_file, 'r'))
        if mix is True:
            splits_eval = splits_eval.split('_',1)[-1]
        splits = [splits_eval]

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




from pcache_fileio import fileio
import io
from dataset.utils import pre_caption
# VG 的box和refcoco一致  xywh
class VG_dataset(Dataset):
    def __init__(self, ann_file_list, transform, image_root, max_words=70):
        # pcache path
        self.image_root = image_root
        self.data = torch.load(ann_file_list[0])
        self.transform = transform
        self.max_words = max_words
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_file, bbox, phrase = self.data[index]
        phrase = pre_caption(phrase, self.max_words)
        bbox=(np.array(bbox)).astype(np.float)
        with open(f'{self.image_root}/visual_genome/{img_file}', "r") as fd:
            image = Image.open(io.BytesIO(fd.read())).convert("RGB")
        target = {}
        target['bbox'] = bbox

        image_iter, target = ConvertCocoPolysToMask(image, target)
        image_iter, target = self.transform(image_iter, target)

        return phrase, image_iter, target['boxes'], index, img_file



# bounding box xywh
class GREC_dataset_Train(Dataset):
    def __init__(self, ann_file_list, transform, image_root, max_words=70):
        self.image_root = image_root

        self.anns = json.load(open(ann_file_list[0], 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        data = self.anns[index]
        bbox = data['bbox']
        sent = data['sent']
        ref_id = data['ref_id']
        empty = data['empty']
        image_file_name = data['img_file_name']
        img_path=os.path.join(self.image_root, image_file_name)
        image = Image.open(img_path).convert('RGB') 

        target = {}
        target['bbox'] = bbox

        image_iter, target = ConvertCocoPolysToMask(image, target, use_keep=(not empty))
        image_iter, target = self.transform(image_iter, target)
        target['empty'] = empty

        return sent, image_iter, target, ref_id, image_file_name


class GREC_dataset_Val(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, splits_eval='val', mix=True):
        self.image_root = image_root

        self.anns = json.load(open(ann_file, 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        data = self.anns[index]
        bbox = data['bbox']
        sent = data['sent']
        ref_id = data['ref_id']
        empty = data['empty']
        image_file_name = data['img_file_name']
        img_path=os.path.join(self.image_root, image_file_name)
        try:
            image = Image.open(img_path).convert('RGB') 
        except:
            img_path = f'{img_path.rsplit("_",1)[0]}.jpg'
            image = Image.open(img_path).convert('RGB') 

        target = {}
        target['bbox'] = bbox

        image_iter, target = ConvertCocoPolysToMask(image, target,use_keep=(not empty))
        image_iter, target = self.transform(image_iter, target)
        target['empty'] = empty

        return sent, image_iter, target, ref_id, image_file_name


