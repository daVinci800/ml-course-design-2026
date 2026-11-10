from functools import lru_cache
from collections import defaultdict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, Sampler
from .geometry import roi_matrix, warp, transform_points, box_mask
from .cache import render_record

MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
STD = np.array([0.229,0.224,0.225],dtype=np.float32)


def stage_settings(config, stage):
    settings = dict(config[stage])
    if config.get('cache_dir'):
        settings['cache_dir'] = config['cache_dir']
        settings['coarse_cache'] = stage=='coarse' and config.get('coarse_cache',False)
    settings['average_duplicate_targets'] = config.get('average_duplicate_targets',False)
    return settings


class EpochSampler(Sampler):
    """Epoch is carried in each index, including with Windows persistent workers."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = self.dataset.epoch
        generator = torch.Generator().manual_seed(self.seed+epoch)
        return iter((epoch,int(i)) for i in torch.randperm(len(self.dataset),generator=generator))


def worker_init(worker_id):
    torch.set_num_threads(1)


@lru_cache(maxsize=8)
def load_image(path):
    with Image.open(path) as im:
        return im.convert('RGB').copy()


def to_tensor(image):
    array = np.asarray(image,dtype=np.float32)/255
    return torch.from_numpy(((array-MEAN)/STD).transpose(2,0,1).copy())


def color_augment(image, rng):
    for transform in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        image = transform(image).enhance(float(rng.uniform(0.8,1.2)))
    if rng.random()<0.15:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2,1.0))))
    return image


class FoveaDataset(Dataset):
    def __init__(self, records, stage, config, training=False, centers=None, seed=0):
        self.records, self.stage, self.config = records, stage, config
        self.training, self.centers = training, centers or {}
        self.seed, self.epoch = seed, 0
        self.target_overrides, self.sample_weights = {}, {}
        if training and config.get('average_duplicate_targets',False):
            buckets = defaultdict(list)
            for r in records:
                buckets[(r.get('decoded_sha256',str(r['id'])),bool(r.get('visible',True)))].append(r)
            for group in buckets.values():
                mean = np.mean([r['xy'] for r in group],axis=0)
                for r in group:
                    self.target_overrides[r['id']] = mean
                    self.sample_weights[r['id']] = 1/len(group)

    def __len__(self):
        return len(self.records) * (self.config.get('repeats',1) if self.training else 1)

    def __getitem__(self, index):
        epoch = self.epoch
        if isinstance(index,tuple):
            epoch,index = index
        r = self.records[index % len(self.records)]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,epoch,index]))
        n = self.config['size']
        xy = np.array(self.target_overrides.get(r['id'],r.get('xy',[0,0])),dtype=np.float64)
        visible = bool(r.get('visible',True))
        if self.stage=='coarse':
            center, side = np.array(r['fov_center']), r['fov_side']
        else:
            side = self.config['crop_side']
            if self.training:
                # Fixed tensor resolution, occasional wider original-pixel view.
                # This supplies context and covers adaptive-window inference.
                if self.config.get('wide_crop_probability',0.0)>0 and rng.random()<self.config['wide_crop_probability']:
                    low,high=self.config.get('wide_crop_scale_range',[1.25,1.5])
                    if low<1 or high<low:
                        raise ValueError('Invalid wide crop scale range.')
                    side*=float(rng.uniform(low,high))
                center = xy.copy() if visible else np.array(r['fov_center'])
                wide = rng.random()<self.config.get('wide_jitter_probability',0.15)
                sigma = self.config.get('wide_jitter_px',125) if wide else self.config.get('jitter_px',55)
                # Bounded shifts keep visible labels in the crop during supervised refinement.
                shift = np.clip(rng.normal(0,sigma,2),-side*0.32,side*0.32)
                center = center + shift
            else:
                if r['id'] not in self.centers:
                    raise ValueError(f'Missing predicted coarse center for id {r["id"]}; validation never uses GT crops.')
                center = np.array(self.centers[r['id']], dtype=float)
        angle, flip = 0,False
        if self.training:
            if self.stage=='coarse':
                center = center + rng.uniform(-0.035,0.035,2)*side
            side *= float(rng.uniform(0.90,1.12))
            angle = float(rng.uniform(-12,12))
            flip = bool(rng.random()<0.5)
        matrix = roi_matrix(center,side,n,angle,flip)
        image = render_record(r,matrix,n,self.config.get('cache_dir'),self.config.get('coarse_cache',False))
        if self.training:
            image = color_augment(image,rng)
        q = transform_points(matrix,xy)
        inside = bool(0<=q[0]<n and 0<=q[1]<n)
        return dict(image=to_tensor(image), target=torch.tensor((q+0.5)/n,dtype=torch.float32),
                    coord_valid=torch.tensor(visible and inside,dtype=torch.float32),
                    presence=torch.tensor(float(visible)),
                    box=torch.from_numpy(box_mask(r.get('box') if visible else None,matrix,n))[None],
                    box_valid=torch.tensor(float(visible and r.get('box') is not None)),
                    matrix=torch.tensor(matrix,dtype=torch.float64),
                    sample_weight=torch.tensor(self.sample_weights.get(r['id'],1.),dtype=torch.float32),
                    side=torch.tensor(side,dtype=torch.float32), id=r['id'])
