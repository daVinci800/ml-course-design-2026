"""Lossless decoded-pixel cache; local crops map only the required source region."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import transform_points, warp, roi_matrix


def cache_path(record, directory):
    return Path(directory) / (record['decoded_sha256'] + '.npy')


def global_matrix(record,size=1024):
    return roi_matrix(record['fov_center'],record['fov_side']*1.5,size)


def global_cache_path(record,directory,size=1024):
    key=hashlib.sha256(global_matrix(record,size).tobytes()).hexdigest()[:12]
    return Path(directory)/(record['decoded_sha256']+f'.global{size}.{key}.npy')


def build_cache(records, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    created, reused, total, global_created = 0, 0, 0, 0
    for record in records:
        path = cache_path(record, directory)
        expected = (record['height'], record['width'], 3)
        if path.exists():
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != expected or array.dtype != np.uint8:
                raise ValueError(f'Invalid pixel cache: {path}')
            del array
            reused += 1
        else:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Source image changed after prepare: {record["path"]}')
            with Image.open(record['path']) as image:
                array = np.asarray(image.convert('RGB'))
            if array.shape != expected or hashlib.sha256(array.tobytes()).hexdigest() != record['decoded_sha256']:
                raise ValueError(f'Decoded pixels differ from manifest: {record["id"]}')
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(path)
            created += 1
        total += path.stat().st_size
        compact=global_cache_path(record,directory)
        if not compact.exists():
            source=np.load(path,mmap_mode='r',allow_pickle=False)
            image=Image.fromarray(source)
            pyramid=np.asarray(warp(image,global_matrix(record),1024))
            temporary=compact.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream,pyramid,allow_pickle=False)
            temporary.replace(compact)
            global_created+=1
    result = dict(created=created, reused=reused, referenced_bytes=total,
                  global_created=global_created,global_size=1024,
                  unique_images=len({r['decoded_sha256'] for r in records}),
                  note='Raw pixels are lossless for refinement. A compact 1024px pyramid speeds global views. No labels are used.')
    (directory/'cache_info.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


@lru_cache(maxsize=32)
def mapped_image(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


@lru_cache(maxsize=64)
def compact_image(path):
    # About 192 MiB per worker at capacity, shared inputs never mutated.
    return Image.fromarray(np.load(path,allow_pickle=False))


def render_record(record, matrix, size, directory=None, coarse=False):
    path = cache_path(record, directory) if directory and 'decoded_sha256' in record else None
    if path is not None and coarse:
        compact=global_cache_path(record,directory)
        try:
            return warp(compact_image(str(compact)),np.asarray(matrix) @ np.linalg.inv(global_matrix(record)),size)
        except FileNotFoundError:
            pass
    try:
        array=mapped_image(str(path)) if path else None
    except FileNotFoundError:
        array=None
    if array is None:
        # Import lazily to avoid a data/cache import cycle.
        from .data import load_image
        return warp(load_image(record['path']), matrix, size)
    matrix=np.asarray(matrix)
    corners = np.array([[-0.5,-0.5],[size-0.5,-0.5],[-0.5,size-0.5],[size-0.5,size-0.5]])
    source = transform_points(np.linalg.inv(matrix), corners)
    lower = np.maximum(np.floor(source.min(0)).astype(int)-2, [0,0])
    upper = np.minimum(np.ceil(source.max(0)).astype(int)+3, [array.shape[1],array.shape[0]])
    if (upper <= lower).any():
        return Image.new('RGB',(size,size))
    x0,y0 = lower
    x1,y1 = upper
    image = Image.fromarray(np.asarray(array[y0:y1,x0:x1]))
    translation = np.eye(3)
    translation[:2,2] = [x0,y0]
    return warp(image, np.asarray(matrix) @ translation, size)
