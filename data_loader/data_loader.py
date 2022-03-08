import sys
import cv2
import torch
from glob import glob
import numpy as np
import torch.utils.data as data
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


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
        self._anno_paths = sorted(glob(f'{root}/*.xml'))
        self._img_paths = sorted(glob(f'{root}/*.jpg'))
        if len(self._img_paths) != len(self._anno_paths):
            raise ValueError(f"Got {len(self._anno_paths)} annotations," +
                             f"{len(self._img_paths)} images")

    def __getitem__(self, idx):
        target = ET.parse(self._anno_paths[idx]).getroot()
        img = cv2.imread(self._img_paths[idx])[:, :, (2, 1, 0)]
        h, w, c = img.shape
        if self.target_transform:
            target = self.target_transform(target, w, h)

        target = np.array(target).astype(np.float32)
        if self.transform:
            boxes, labels = (target[:, :4], target[:, 4]) if len(target) > 0 else ([], [])
            img, boxes, labels = self.transform(img, boxes, labels)
            if len(target) > 0:
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        return img, target

    def __len__(self):
        return len(self._img_paths)
