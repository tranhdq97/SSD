import torch
import cv2
import numpy as np
from numpy import random


class Compose:
    """Composes several augmentation

    Args:
        transforms (List[Transform]): list of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)

        return img, boxes, labels


class ToTensor:
    """Transforms cv2 image to tensor"""
    def __call__(self, img, boxes=None, labels=None):
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), boxes, labels


class ToImage:
    """Transforms tensor to cv2 image"""
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToAbsoluteCoords:
    """Transforms percent coords to absolute coords"""
    def __call__(self, img, boxes, labels=None):
        h, w, _ = img.shape
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        return img, boxes, labels


class ToPercentCoords:
    """Transforms absolute coords to percent coords"""
    def __call__(self, img, boxes, labels=None):
        h, w, _ = img.shape
        if len(boxes) > 0:
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
        return img, boxes, labels


class Resize:
    """Resizes image and bounding boxes"""
    def __init__(self, size=300):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size))
        return img, boxes, labels


class RandomHFlip:
    """Randomly flips image and bounding boxes"""
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img = img[:, ::-1]
            if len(boxes) > 0:
                boxes[:, 0::2] = 1 - boxes[:, 2::-2]

        return img, boxes, labels


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Augmentation:
    """Default augmentation"""
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.augment = Compose([
            # RandomHFlip(),
            Resize(size),
            # SubtractMeans(mean),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
