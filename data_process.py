

import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import numpy as np  
from torchvision.transforms import functional as F

import json, re, random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

import torch

import torchvision

import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm


# train 80512  test 9602

lmdb_dir = '/home/ma-user/work/workspace/code/vg_bridge/datasets/lmdb/refcoco/val.lmdb'
lmdb_dir_full = '/home/ma-user/work/workspace/code/vg_bridge/datasets/lmdb/refcoco/val_full.lmdb'


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

env = lmdb.open(lmdb_dir,
                subdir=os.path.isdir(lmdb_dir),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)


print("Generate LMDB to %s" % lmdb_dir_full)
isdir = os.path.isdir(lmdb_dir_full)
db = lmdb.open(lmdb_dir_full, subdir=isdir,
                map_size=1099511627776 * 2, readonly=False,
                meminit=False, map_async=True)
txn = db.begin(write=True)

# dict_keys(['img', 'mask', 'cat', 'seg_id', 'img_name', 'num_sents', 'sents'])
key_ls = []
with env.begin(write=False) as reader:
    keys = loads_pyarrow(reader.get(b'__keys__'))
    for key in tqdm(keys):
        byteflow = reader.get(key)
        ref = loads_pyarrow(byteflow)
        sents = ref['sents']
        img, cat, seg_id, img_name, mask = ref['img'], ref['cat'], ref['seg_id'], ref['img_name'], ref['mask']
        for i, sent in enumerate(sents):
            idx = key.decode('ascii') + f'_{i}'
            key_ls.append(idx)
            data = {'img': img, 'mask': mask, 'cat': cat,
                'seg_id': seg_id, 'img_name': img_name, 'sent': sent}
            txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(data))
                    # print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    _keys = [u'{}'.format(k).encode('ascii') for k in key_ls]
    print(len(_keys))
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(_keys))
        txn.put(b'__len__', dumps_pyarrow(len(_keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


        

