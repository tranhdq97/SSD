import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function"""
    def __init__(self, cfg, overlap_thresh, neg_pos, device):
        super().__init__()
        self.device = device
        self.num_classes = cfg['num_classes']
        self.variance = cfg['variance']
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        bs = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        loc_t = torch.Tensor(bs, num_priors, 4)
        conf_t = torch.LongTensor(bs, num_priors)
        for idx in range(bs):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            # defaults = priors.data
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)

        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,
                                                                            1))
        # Hard Negative Mining
        loss_c = loss_c.view(bs, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1,
                                                           self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
