import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.augmentation import Augmentation
from data_loader.data_loader import VOCLoader, VOCAnnotationTransform
from model.model import build_ssd
from model.loss import MultiBoxLoss
from trainer.trainer import Trainer
from configs.config import VOC, MEANS
from utils.util import my_collate


if __name__ == '__main__':
    # Device
    cfg = VOC
    lr = 1e-3
    gamma = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    scheduler_gamma = 0.5
    num_workers = 1
    batch_size = 32
    epochs = 100000
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    # Dataset
    train_dataset = VOCLoader(root='data',
                              transform=Augmentation(size=cfg['min_dim'],
                                                     mean=MEANS),
                              target_transform=VOCAnnotationTransform())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False,
                              pin_memory=True, num_workers=num_workers,
                              collate_fn=my_collate)
    valid_loader = DataLoader(train_dataset, batch_size=3, shuffle=False,
                              pin_memory=True, num_workers=num_workers,
                              collate_fn=my_collate)

    # Network
    model = build_ssd(size=cfg['min_dim'], cfg=cfg,
                      num_classes=cfg['num_classes'])

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=scheduler_gamma,
                       verbose=False)

    # Loss
    loss = MultiBoxLoss(cfg=cfg,
                        overlap_thresh=0.5,
                        neg_pos=3,
                        device=device)

    # Metric

    # Training
    trainer = Trainer(model=model, device=device, train_loader=train_loader,
                      valid_loader=valid_loader, loss=loss, cfg=cfg,
                      optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                      metric=None)

    trainer.train()
