import torch
import torch.nn as nn
import torch.nn.functional as F


class SSD(nn.Module):
    """Single Shot Multi-box

    Args:
        phase (string): 'train' or 'test mode
        size (int): input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multi-box loc and conf layers
        head: 'mutil-box head' consists of loc and cof conv layers
    """
    def __init__(self, phase, size, base, extras, head, num_classes):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        # self.cfg = (coco, voc)[num_classes == 21]

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm

