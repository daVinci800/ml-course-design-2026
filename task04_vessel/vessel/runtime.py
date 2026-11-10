import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .model import build_model


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config['in_channels'] not in [3,4] or config['batch_size'] < 1:
        raise ValueError('Invalid config')
    return config


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install the CUDA PyTorch build; see README.')
    return device


def training_fingerprint(data_root):
    root=Path(data_root)
    digest=hashlib.sha256()
    # Test set is deliberately excluded from all training/checkpoint decisions.
    for folder in ['train/image','train/label']:
        for path in sorted((root/folder).glob('*.jpg'),key=lambda p:int(p.stem)):
            digest.update(f'{folder}/{path.name}'.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def save_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def checkpoint_load(path, device):
    # Load only trusted, locally produced checkpoints. Training metadata contains RNG states.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('completed_epochs',0) < 1:
        raise ValueError('Checkpoint has no completed training epoch')
    model=build_model(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model,checkpoint


def atomic_torch_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp); tmp.replace(path)
