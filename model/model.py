import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from configs.config import VOC
from .modules.l2norm import L2Norm
from .functions.detection import Detection
from .functions.prior_box import PriorBox
from utils.model import multi_box, vgg_layers, add_extras
from configs.config import BASE, EXTRAS, MBOX


class SSD(nn.Module):
    """Single Shot Multi-box

    Args:
        phase (string): 'train' or 'test mode
        size (int): input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multi-box loc and conf layers
        head: 'mutil-box head' consists of loc and cof conv layers
    """

    def __init__(self, size, base, extras, head, num_classes, cfg):
        super().__init__()
        self.phase = 'test'
        self.num_classes = num_classes
        self.size = size
        self.cfg = cfg
        self.priors = Variable(PriorBox(self.cfg).forward())
        # SSD network
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detection(num_classes, 0, 200, 0.01, 0.45, self.cfg)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x

        Args:
            x (tensor | bs, 3, 300, 300)
        """
        sources, loc, conf = [], [], []
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.extend([s, x])
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),
                self.priors.type(type(x.data)).to(x.device)
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors.to(x.device)
            )
        return output

    def to_train(self):
        self.train()
        self.phase = 'train'

    def to_eval(self):
        self.eval()
        self.phase = 'test'


def build_ssd(size=300, num_classes=4, cfg=VOC):
    if size != 300:
        raise ValueError("Currently, only size 300 is supported")

    base_, extras_, head_ = multi_box(vgg_layers(BASE[str(size)], 3),
                                      add_extras(EXTRAS[str(size)], 1024),
                                      MBOX[str(size)],
                                      num_classes)
    return SSD(size, base_, extras_, head_, num_classes, cfg)
