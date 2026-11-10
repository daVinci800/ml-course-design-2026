"""Exact official pixel semantics: 512x512, Pillow BILINEAR, >0, row major."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image


def encode_rle(mask):
    flat = np.asarray(mask,dtype=bool).ravel(order='C').astype(np.int8)
    changes = np.flatnonzero(np.diff(np.pad(flat,(1,1))))+1
    changes[1::2] -= changes[::2]
    return ' '.join(map(str,changes.tolist()))


def decode_rle(text, shape=(512,512)):
    tokens = text.strip().split()
    if len(tokens)%2:
        raise ValueError('Odd number of RLE tokens')
    result = np.zeros(int(np.prod(shape)),dtype=bool)
    previous_end = 0
    for s,l in zip(tokens[::2],tokens[1::2]):
        start,length = int(s),int(l)
        if start<=0 or length<=0 or start<=previous_end or start+length-1>result.size:
            raise ValueError('RLE must be positive, sorted, nonoverlapping, and in bounds')
        result[start-1:start-1+length] = True
        previous_end = start+length-1
    return result.reshape(shape)


def official_foreground(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L').resize((512,512),Image.Resampling.BILINEAR))>0


def resize_probability(prob):
    return np.asarray(Image.fromarray(np.asarray(prob,dtype=np.float32)).resize((512,512),Image.Resampling.BILINEAR))


def read_ids(sample):
    with open(sample,newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['Id','Predicted']:
            raise ValueError('Sample CSV must contain exactly Id,Predicted')
        ids = [r['Id'] for r in reader]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Empty or duplicate sample IDs')
    return ids


def masks_to_csv(mask_dir,sample,out,require_512=True):
    ids = read_ids(sample)
    paths = {p.stem:p for p in Path(mask_dir).glob('*.png')}
    if set(paths)!=set(ids):
        raise ValueError(f'Mask IDs differ from sample: missing={set(ids)-set(paths)}, extra={set(paths)-set(ids)}')
    rows=[]
    for sid in ids:
        with Image.open(paths[sid]) as im:
            a=np.asarray(im.convert('L'))
            if not set(np.unique(a)).issubset({0,255}):
                raise ValueError(f'{sid}: output must be lossless binary PNG')
            if require_512 and im.size!=(512,512):
                raise ValueError(f'{sid}: use explicit 512x512 submission masks')
        foreground=official_foreground(paths[sid])
        encoded=encode_rle(foreground)
        if not np.array_equal(decode_rle(encoded),foreground):
            raise AssertionError('RLE round trip failed')
        rows.append({'Id':sid,'Predicted':encoded})
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['Id','Predicted'])
        writer.writeheader(); writer.writerows(rows)
    return rows
