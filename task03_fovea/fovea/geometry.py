"""Pixel-center geometry shared by augmentation, training, TTA and inference."""
import numpy as np
from PIL import Image


def transform_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    return (np.c_[points.reshape(-1, 2), np.ones(points.size // 2)] @ np.asarray(matrix).T)[:, :2].reshape(points.shape)


def roi_matrix(center, side, size, angle=0.0, flip=False):
    theta = np.deg2rad(angle)
    linear = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if flip:
        linear = np.diag([-1.0, 1.0]) @ linear
    linear *= size / side
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = (size - 1) / 2 - linear @ np.asarray(center)
    return matrix


def warp(image, matrix, size):
    inverse = np.linalg.inv(matrix)
    # Pillow evaluates its affine at pixel edge coordinates (center = index + .5).
    offset = inverse[:2, 2] + 0.5 - inverse[:2, :2] @ np.array([0.5, 0.5])
    coeff = (*inverse[0, :2], offset[0], *inverse[1, :2], offset[1])
    return image.transform((size, size), Image.Transform.AFFINE, coeff,
                           resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def fov_roi(image):
    small = np.asarray(image.resize((256, 256)), dtype=np.uint8)
    mask = small.max(axis=2) > 12
    xs = np.flatnonzero(mask.sum(axis=0) >= 5)
    ys = np.flatnonzero(mask.sum(axis=1) >= 5)
    w, h = image.size
    if len(xs) < 20 or len(ys) < 20:
        return [(w - 1) / 2, (h - 1) / 2], float(max(w, h))
    x0, x1 = xs[0] * w / 256, (xs[-1] + 1) * w / 256
    y0, y1 = ys[0] * h / 256, (ys[-1] + 1) * h / 256
    return [(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2], max(x1 - x0, y1 - y0) * 1.03


def decode_norm(norm_xy, matrix, size):
    return transform_points(np.linalg.inv(matrix), np.asarray(norm_xy) * size - 0.5)


def box_mask(box, matrix, size, stride=4):
    h = size // stride
    if not box:
        return np.zeros((h, h), dtype=np.float32)
    y, x = np.mgrid[:h, :h]
    points = np.stack(((x + 0.5) * stride - 0.5, (y + 0.5) * stride - 0.5), axis=-1)
    original = transform_points(np.linalg.inv(matrix), points)
    return ((original[..., 0] >= box[0]) & (original[..., 0] <= box[2]) &
            (original[..., 1] >= box[1]) & (original[..., 1] <= box[3])).astype(np.float32)
