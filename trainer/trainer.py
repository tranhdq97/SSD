import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from configs.config import VOC_COLORS


class Trainer:
    """Training module"""
    def __init__(self, model, device, train_loader, valid_loader, loss,
                 optimizer, scheduler, epochs, metric, cfg):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.iter = 0
        self.loc_loss = 0
        self.conf_loss = 0
        self.epochs = epochs
        self.metric = metric

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            # self.scheduler.step()
            if epoch % 1000 == 0:
                print(100 * '-', '\n\nEvaluating: ', epoch)
                self.eval_one_epoch(epoch)

            if epoch % 5000 == 0:
                torch.save(self.model.state_dict(), f'saved/models/{epoch}.pth')

    def train_one_epoch(self, epoch):
        tqdm_loader = tqdm(self.train_loader)
        self.model.to_train()
        for b_idx, batch in enumerate(tqdm_loader):
            loss, loss_l, loss_c = self.train_step(batch)
            self.iter += 1
            tqdm_loader.set_description(
                f'Epoch: {epoch}/{self.epochs} | Iter: {self.iter} | Loss: ' +
                f'total: {loss:.5f} - loc: {loss_l:.5f} - conf: {loss_c:.5f}'
            )

    def eval_one_epoch(self, epoch):
        """
        * Note: draw detected bounding boxes into images for visualizing the
        training results
        TODO: have to implement evaluating metric.
        """
        tqdm_loader = tqdm(self.valid_loader)
        self.model.to_eval()
        for b_idx, batch in enumerate(tqdm_loader):
            images, targets, hw = self.prepare_data(batch)
            with torch.no_grad():
                out = self.model(images)

            bs, num_cls, _, _ = out.shape
            for i in range(bs):
                h, w = hw[i]
                # images
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = img.astype(np.uint8)
                img = cv2.resize(img, (w, h))
                # boxes
                for j in range(1, num_cls):
                    dets = out[i, j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue

                    boxes = dets[:, 1:]
                    boxes[:, [0, 2]] *= w
                    boxes[:, [1, 3]] *= h
                    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0,
                                                   max=w)
                    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0,
                                                   max=h)
                    scores = dets[:, 0].cpu().numpy()
                    boxes = boxes.cpu().numpy().astype(np.int)
                    for idx, score in enumerate(scores):
                        x1, y1, x2, y2 = boxes[idx]
                        cv2.rectangle(img, (x1, y1), (x2, y2), VOC_COLORS[j-1],
                                      3)

                cv2.imwrite(f'saved/{epoch}_{i}.jpg', img)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        images, targets, _ = self.prepare_data(batch)
        out = self.model(images)
        loss_l, loss_c = self.loss(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        self.optimizer.step()
        self.loc_loss += loss_l.data
        self.conf_loss += loss_c.data
        return loss.item(), loss_l.item(), loss_c.item()

    def prepare_data(self, batch):
        images, targets, hw = batch
        images = images.to(self.device)
        targets = [Variable(anno.to(self.device)) for anno in
                   targets]
        return images, targets, hw
