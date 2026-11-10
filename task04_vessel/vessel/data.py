import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def load_label(path):
    # Actual supplied labels: white vessels, black background.
    with Image.open(path) as image:
        return (np.asarray(image.convert('L')) >= 128).astype(np.float32)


def preprocess(rgb, config):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    channels = [rgb.astype(np.float32)/127.5 - 1.0]
    if config['in_channels'] == 4:
        clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(config['clahe_grid'],)*2)
        enhanced = clahe.apply(rgb[...,1]).astype(np.float32)/127.5 - 1.0
        channels.append(enhanced[...,None])
    elif config['in_channels'] != 3:
        raise ValueError('in_channels must be 3 (RGB) or 4 (RGB + green CLAHE)')
    return torch.from_numpy(np.concatenate(channels, -1).transpose(2,0,1).copy())


def augment(rgb, mask, rng):
    # All spatial transforms are paired. Full-resolution images retain global context.
    k = int(rng.integers(4))
    rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        rgb, mask = rgb[:,::-1], mask[:,::-1]
    # A fixed square canvas avoids odd-rotation shape differences in a batch.
    h, w = mask.shape
    size = ((max(h,w) + 15)//16)*16
    top, left = (size-h)//2, (size-w)//2
    rgb = cv2.copyMakeBorder(np.ascontiguousarray(rgb), top, size-h-top, left, size-w-left, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.copyMakeBorder(np.ascontiguousarray(mask), top, size-h-top, left, size-w-left, cv2.BORDER_CONSTANT, value=0)
    if rng.random() < 0.8:
        matrix = cv2.getRotationMatrix2D((size/2,size/2), float(rng.uniform(-15,15)), float(rng.uniform(0.9,1.1)))
        matrix[:,2] += rng.uniform(-0.035,0.035,2)*size
        rgb = cv2.warpAffine(rgb, matrix, (size,size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, matrix, (size,size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.8:
        values = rgb.astype(np.float32)/255
        values = np.power(values, float(rng.uniform(0.8,1.25)))
        values *= rng.uniform(0.9,1.1,(1,1,3))
        values = (values-0.5)*float(rng.uniform(0.9,1.1))+0.5
        rgb = (np.clip(values,0,1)*255).astype(np.uint8)
    if rng.random() < 0.15:
        noise = rng.normal(0, rng.uniform(1,4), rgb.shape)
        rgb = np.clip(rgb.astype(np.float32)+noise,0,255).astype(np.uint8)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)


class TrainDataset(Dataset):
    def __init__(self, root, ids, config):
        root = Path(root)
        self.config = config
        self.items = [(load_rgb(root/f'train/image/{sid}.jpg'), load_label(root/f'train/label/{sid}.jpg')) for sid in ids]
        self.epoch = 0

    def __len__(self):
        return self.config['samples_per_epoch']

    def __getitem__(self, index):
        # Deterministic across workers and resumptions; worker count does not change examples.
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index]))
        rgb, mask = self.items[int(rng.integers(len(self.items)))]
        rgb, mask = augment(rgb.copy(), mask.copy(), rng)
        return preprocess(rgb,self.config), torch.from_numpy(mask[None].copy())


def make_folds(root, k=5, seed=20261107):
    ids = sorted([p.stem for p in (Path(root)/'train/image').glob('*.jpg')], key=int)
    if len(ids) < k or k < 2:
        raise ValueError('Need at least k images and k >= 2')
    # Only train labels influence balanced fold construction. Never inspect test labels.
    ranked = sorted(ids, key=lambda sid: float(load_label(Path(root)/f'train/label/{sid}.jpg').mean()))
    buckets = [[] for _ in range(k)]
    rng = np.random.default_rng(seed)
    for offset in range(0,len(ranked),k):
        for fold, sid in zip(rng.permutation(k), ranked[offset:offset+k]):
            buckets[int(fold)].append(sid)
    folds = [{'fold':f,'train_ids':[sid for sid in ids if sid not in val],'val_ids':sorted(val,key=int)} for f,val in enumerate(buckets)]
    # Byte-identical training images must never cross folds.
    hashes = [hashlib.sha256((Path(root)/f'train/image/{sid}.jpg').read_bytes()).hexdigest() for sid in ids]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate training images detected; group duplicates before creating folds')
    return {'seed':seed,'fold_count':k,'folds':folds, 'unit':'whole image; no patch-level split', 'patient_grouping':'No patient IDs supplied. Different images of the same patient cannot be ruled out.'}


def read_folds(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
