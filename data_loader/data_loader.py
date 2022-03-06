import sys
import os
import cv2
import torch
from glob import glob
import numpy as np
import torch.utils.data as data
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementPath as ET


VOC_CLASSES = ('human', 'mask', 'no_mask')


class VOCAnnotationTransform:
    """Transforms a VOC annotation into a Tensor of bbox coords and label index

    Args:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
    """
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES,
                                                     range(len(VOC_CLASSES))))

    def __call__(self, target, width, height):
        """
        Args:
            target: the target annotation

        Returns:
            A list containing lists of bounding boxes [bbox coords, classname]
        """
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            bndbox.append(self.class_to_ind[name])
            res.append(bndbox)

        return res


class VOCLoader(data.Dataset):
    """VOC Dataset

    Args:
        root (str): filepath to VOC folder
        transform (callable, optional): transformation to perform on the input
            image
        target_transform (callable, optional): transformation to perform on the
            target annotation
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=VOCAnnotationTransform()):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._anno_paths = os.path.join()
        pass


    def __getitem__(self, item):

        return

