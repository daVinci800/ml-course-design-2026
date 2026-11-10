"""Coordinate supervision for a frozen detector's single-keypoint branch."""
import torch
from torch.nn import functional as F


def freeze_batchnorm_stats(model):
    for layer in model.modules():
        if isinstance(layer,torch.nn.BatchNorm2d): layer.eval()


class CoordinateLoss(torch.nn.Module):
    """Uncapped robust coordinate loss; avoids single-keypoint OKS saturation."""
    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        scale = area.detach().clamp_min(1).sqrt().reshape(-1, 1, 1) * .1
        error = (pred_kpts[..., :2].float() - gt_kpts[..., :2].float()) / scale
        per_point = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
        return (per_point * kpt_mask).sum() / kpt_mask.sum().clamp_min(1)


def extract_points(results, records, fallback):
    import numpy as np
    if len(results) != len(records): raise ValueError('Prediction count mismatch')
    output = {}
    for result, record in zip(results, records):
        xy = None
        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            scores = result.boxes.conf.detach().float().cpu().numpy()
            points = result.keypoints.xy.detach().float().cpu().numpy()[:, 0]
            valid = np.isfinite(points).all(1) & np.isfinite(scores)
            if valid.any():
                index = np.where(valid)[0][np.argmax(scores[valid])]
                xy = points[index]
        missing = xy is None
        if missing: xy = fallback[record['id']]['xy']
        xy = np.clip(xy, [0, 0], [record['width']-1, record['height']-1])
        output[record['id']] = dict(xy=xy.tolist(), detector_miss=missing)
    return output
