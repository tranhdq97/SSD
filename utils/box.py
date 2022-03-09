import os.path

import numpy as np
import torch
from configs.config import VOC_CLASSES


def to_point_form(boxes):
    """Converts prior_boxes to (xmin, ymin, xmax, ymax)

    Args:
        boxes (tensor): center-size default boxes from prior_box layers

    Returns:
        boxes (tensor): converted (xmin, ymin, xmax, ymax) form of boxes
    """
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1
    )

#
# def to_center_size(boxes):
#     """Converts prior_boxes to (cx, cy, w, h)
#
#     Args:
#         boxes (tensor): point_form boxes
#
#     Returns:
#         boxes (tensor): converted (cx, cy, w, h) form of boxes
#     """
#     return torch.cat(
#         ((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]), 1
#     )


def intersect(box_a, box_b):
    """Computes the area of intersect between box_a and box_b

    Args:
        box_a (tensor | A,4): bounding boxes
        box_b (tensor | B,4): bounding boxes

    Returns:
        (tensor | A,B) intersection area
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Computes the jaccard overlap of two sets of boxes

    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        box_a (tensor | num_objects,4): ground truth bounding boxes
        box_b (tensor | num_objects,4): prior boxes from prior_box layers

    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Matches each prior box with the ground truth box of the highest jaccard
    overlap. encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location predictions.

    Args:
        threshold (float): the overlap threshold used when matching boxes
        truths (tensor | num_obj, num_priors): ground-truth boxes
        priors (tensor | n_priors,4): prior boxes from prior_box layers
        variances (tensor | num_prior,4): variances corresponding to each prior
            coord
        labels (tensor | num_obj): all the class labels for the image
        loc_t (tensor): tensor to be filled with encoded location targets
        conf_t (tensor): tensor to be filled with matched indices for conf preds
        idx (int): current batch index

    Returns:
        the matched indices corresponding to 1)location and 2)confidence preds
    """
    overlaps = jaccard(truths, to_point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def encode(matched, priors, variances):
    """Encodes the variances from the prior_box layers into the ground truth
    boxes we have matched with the prior boxes

    Args:
        matched (tensor | num_priors,4): coords of ground truth for each prior
            in point form
        priors (tensor | num_priors,4): prior boxes in center-offset form
        variances (list[float]): variances of prior boxes

    Returns:
        encoded boxes (tensor) Shape [num_priors,4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    """Decodes locations from predictions using priors to undo the encoding
    we did for offset regression at train time

    Args:
        loc (tensor | num_priors,4): location predictions from loc layers
        priors (tensor | num_priors,4): prior boxes in center-offset form
        variances (list[float]): variances of prior boxes

    Returns:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Applies non-maximum suppression at test time to avoid detecting to many
    overlapping bounding boxes for a given object

    Args:
        boxes (tensor | num_priors,4): the location preds for the img
        scores (tensor | num_priors): the class predscores for the img
        overlap (float): the overlap threshold for suppressing unnecessary boxes
        top_k (int): the maximum number of box preds to consider

    Returns:
        the indices of the kept boxes with respect to num_priors
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    # indices of top-k the largest values
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break

        idx = idx[:-1]
        # load bboxes of next highest values
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, min=x2[i])
        yy2 = torch.clamp(yy2, min=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        remaining_areas = torch.index_select(area, 0, idx)
        union = (remaining_areas - inter) + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]

    return keep, count
