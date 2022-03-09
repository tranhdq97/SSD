import torch
from torch.autograd import Function
from utils.box import decode, nms


class Detection(Function):
    """Detect is the final layer of SSD"""
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh,
                 cfg):
        self.num_classes = num_classes
        self.bkg_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative')

        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data (tensor | bs, num_priors * 4): Location predictions from
                loc layers
            conf_data (tensor | bs * num_priors, num_classes): confidence
                predictions from conf layers
            prior_data (tensor | num_priors, 4): prior boxes and variances
                from prior_box layers
        """
        bs = loc_data.size(0)
        num_priors = prior_data.size(0)  # TODO: check with doc
        output = torch.zeros(bs, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(bs, num_priors,
                                    self.num_classes).transpose(2, 1)
        for i in range(bs):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(bs, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
