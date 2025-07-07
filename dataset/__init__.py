import os
import torch
from torch.utils.data import DataLoader

from torchvision.transforms import InterpolationMode

def build_tokenizer(text_encoder: str):
    tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer



from dataset.dataset_grounding import gd_eval_dataset, gd_train_dataset_full, VG_dataset, Flickr30k_dataset, referIt_dataset

from dataset.randaugment import RandomAugment

from transformers import XLMRobertaTokenizer

import clip

import dataset.transforms as T
from torchvision import transforms


def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    # DERT
    DETR_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    DETR_train_transform = T.Compose([
            # T.RandomHorizontalFlip(), 
            # T.RandomResize(scales, max_size=1333),
            # T.ColorJitter(),
            T.Resize([config['image_res'], config['image_res']]),
            DETR_normalize,
        ])
    

    DETR_test_transform = T.Compose([
            T.Resize([config['image_res'], config['image_res']]),
            DETR_normalize,
        ])
    



    # 图像处理
    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])


    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'refcoco':
        
        # train_transform
        train_dataset = gd_train_dataset_full(config['train_file'], DETR_train_transform, config['image_root'])
        # test_transform
        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = gd_eval_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = gd_eval_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)

        return train_dataset, val_dataset_dict, test_dataset_dict
    
    elif dataset == 'VG':
        train_dataset = VG_dataset(config['train_file'], DETR_train_transform, config['image_root'])
        return train_dataset
    
    elif dataset == 'flickr30k':
        train_dataset = Flickr30k_dataset(config['train_file'], DETR_train_transform, config['image_root'])
        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = Flickr30k_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = Flickr30k_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)

        return train_dataset, val_dataset_dict, test_dataset_dict
    
    elif dataset == 'referIt':
        train_dataset = referIt_dataset(config['train_file'], DETR_train_transform, config['image_root'])

        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = referIt_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = referIt_dataset(rpath, DETR_test_transform, config['image_root'], splits_eval=k)


        return train_dataset, val_dataset_dict, test_dataset_dict

    elif dataset == 'seg':

        train_dataset = seg_dataset(lmdb_dir=config['train_lmdb'], mask_dir=config['mask_root'], split=config['train_split'], dataset=config['dataset'], input_size=config['image_res'])
        val_dataset = seg_dataset(lmdb_dir=config['val_lmdb'], mask_dir=config['mask_root'], split=config['val_split'], input_size=config['image_res'], dataset=config['dataset'], mode='val')
        test_dataset = seg_dataset(lmdb_dir=config['test_lmdb'], mask_dir=config['mask_root'], split=config['test_split'], input_size=config['image_res'], dataset=config['dataset'], mode='test')

        return train_dataset, val_dataset, test_dataset
    
 
    
    else:
        raise NotImplementedError(f"dataset == {dataset}")




def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
