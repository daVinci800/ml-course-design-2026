import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .geometry import fov_roi

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(root, destination, seed=20261020, n_folds=5, groups_csv=None):
    root, destination = Path(root).resolve(), Path(destination)
    if not (root / 'train').exists() and (root / 'detection').exists():
        root = root / 'detection'
    if n_folds < 2:
        raise ValueError('At least two folds are required.')
    labels = list(csv.DictReader((root / 'fovea_localization_train_GT.csv').open(encoding='utf-8-sig', newline='')))
    gt = {}
    for row in labels:
        ident = int(Path(row['data']).stem)
        if ident in gt:
            raise ValueError(f'Duplicate GT id {ident}')
        gt[ident] = [float(row['Fovea_X']), float(row['Fovea_Y'])]
    groups = {}
    if groups_csv:
        for row in csv.DictReader(Path(groups_csv).open(encoding='utf-8-sig', newline='')):
            groups[int(row['id'])] = str(row['group'])
    records = []
    seen_ids = set()
    for split in ('train', 'test'):
        for path in sorted((root / split).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            ident = int(path.stem)
            if ident in seen_ids:
                raise ValueError(f'Duplicate image id {ident}')
            seen_ids.add(ident)
            with Image.open(path) as image:
                image = image.convert('RGB')
                w, h = image.size
                center, side = fov_roi(image)
                decoded_sha = hashlib.sha256(image.tobytes()).hexdigest()
            record = dict(id=ident, split=split, path=str(path), width=w, height=h,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), decoded_sha256=decoded_sha,
                          fov_center=center, fov_side=side, fold=-1, box=None)
            if split == 'train':
                xy = gt[ident]
                if not np.isfinite(xy).all() or min(xy) < 0:
                    raise ValueError(f'Invalid GT: {ident}: {xy}')
                visible = xy != [0.0, 0.0]
                if visible and not (xy[0] < w and xy[1] < h):
                    raise ValueError(f'GT outside image: {ident}')
                record.update(xy=xy, visible=visible)
                xml_path = root / 'train_location' / (path.stem + '.xml')
                if xml_path.exists():
                    xml = ET.parse(xml_path).getroot()
                    xml_size = [int(xml.findtext('size/width')), int(xml.findtext('size/height'))]
                    if xml_size != [w, h]:
                        raise ValueError(f'XML size mismatch: {ident}: {xml_size} != {[w,h]}')
                    objects = [o for o in xml.findall('object') if 'fovea' in o.findtext('name', '').lower()]
                    if len(objects) > 1:
                        raise ValueError(f'Multiple fovea boxes in {ident}')
                    if objects:
                        box = [float(objects[0].findtext('bndbox/' + k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                            raise ValueError(f'Invalid box: {ident}')
                        record['box'] = box
            records.append(record)
    train = [r for r in records if r['split'] == 'train']
    if {r['id'] for r in train} != set(gt):
        raise ValueError('Image IDs and GT IDs do not match.')
    # Union-find joins decoded duplicates AND any supplied patient groups.
    parent = {r['id']: r['id'] for r in train}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(b)] = find(a)
    hashes, patients = {}, {}
    for r in train:
        key = r['decoded_sha256']
        if key in hashes:
            union(r['id'], hashes[key])
        hashes[key] = r['id']
        if r['id'] in groups:
            g = groups[r['id']]
            if g in patients:
                union(r['id'], patients[g])
            patients[g] = r['id']
    buckets = defaultdict(list)
    for r in train:
        r['group'] = str(find(r['id']))
        # Camera and visibility balance; no unreliable inferred patient labels.
        r['stratum'] = f"{r['width']}x{r['height']}_{int(r['visible'])}"
        buckets[r['group']].append(r)
    if len(buckets) < n_folds:
        raise ValueError('Too few independent groups for the requested fold count.')
    rng = np.random.default_rng(seed)
    group_list = list(buckets.values())
    rng.shuffle(group_list)
    frequencies = Counter(r['stratum'] for r in train)
    group_list.sort(key=lambda g: (min(frequencies[r['stratum']] for r in g), -len(g)))
    fold_counts = np.zeros(n_folds, dtype=int)
    strata = defaultdict(lambda: np.zeros(n_folds, dtype=int))
    for group in group_list:
        score = fold_counts.astype(float) + rng.uniform(0, 0.01, n_folds)
        for r in group:
            score += 100 * strata[r['stratum']]
        fold = int(score.argmin())
        for r in group:
            r['fold'] = fold
            fold_counts[fold] += 1
            strata[r['stratum']][fold] += 1
    if min(fold_counts) == 0:
        raise ValueError('Empty validation fold.')
    template_path = root / 'sample_submission.csv'
    template = list(csv.DictReader(template_path.open(encoding='utf-8-sig', newline='')))
    expected = {f"{r['id']}_Fovea_{axis}" for r in records if r['split'] == 'test' for axis in ('X','Y')}
    if len(template) != len(expected) or {r['ImageID'] for r in template} != expected:
        raise ValueError('Submission template IDs do not match test images.')
    document = dict(version=1, root=str(root), seed=seed, n_folds=n_folds,
                    template=str(template_path), template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(), records=records)
    write_json(destination / 'manifest.json', document)
    with (destination / 'folds.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','group','stratum','fold'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in train)
    coords = np.array([r['xy'] for r in train], dtype=float)
    visible = np.array([r['visible'] for r in train])
    sizes = np.array([[r['width'], r['height']] for r in train], dtype=float)
    normalized = coords / sizes
    loo = ((normalized.sum(0) - normalized) / (len(train) - 1)) * sizes
    grouped_mean = np.empty_like(coords)
    fov_mean = np.empty_like(coords)
    centers = np.array([r['fov_center'] for r in train])
    sides = np.array([r['fov_side'] for r in train])[:,None]
    fov_relative = (coords-centers)/sides
    for r in train:
        i = train.index(r)
        others = [j for j,s in enumerate(train) if s['fold'] != r['fold']]
        grouped_mean[i] = normalized[others].mean(0) * sizes[i]
        fov_mean[i] = centers[i]+fov_relative[others].mean(0)*sides[i]
    boxed = [r for r in train if r['visible'] and r['box']]
    errors = np.array([[(r['box'][0]+r['box'][2])/2-r['xy'][0],
                        (r['box'][1]+r['box'][3])/2-r['xy'][1]] for r in boxed])
    repeated = [g for g in buckets.values() if len(g)>1]
    cross = [[a['id'], b['id']] for a in train for b in records
             if b['split']=='test' and a['decoded_sha256']==b['decoded_sha256']]
    audit = dict(train=len(train), test=len(records)-len(train), visible=int(visible.sum()),
                 absent_ids=[r['id'] for r in train if not r['visible']],
                 sizes=dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
                 folds=fold_counts.tolist(),
                 grouped_duplicate_ids=[[r['id'] for r in g] for g in repeated],
                 exact_train_test_duplicates=cross,
                 normalized_mean_loo_rmse=float(np.sqrt(np.mean((loo-coords)**2))),
                 normalized_mean_grouped_oof_rmse=float(np.sqrt(np.mean((grouped_mean-coords)**2))),
                 fov_center_rmse=float(np.sqrt(np.mean((centers-coords)**2))),
                 fov_relative_grouped_mean_rmse=float(np.sqrt(np.mean((fov_mean-coords)**2))),
                 oracle_box_center_rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                 warnings=['Patient IDs are unavailable. Exact duplicate grouping cannot exclude all patient-level dependence.',
                           'Oracle box-center error uses ground-truth boxes and is NOT a prediction score.'])
    write_json(destination / 'audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return document
