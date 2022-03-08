import torch
from tqdm import tqdm
from torch.autograd import Variable


class Trainer:
    """Training module"""
    def __init__(self, model, device, train_loader, valid_loader, loss,
                 optimizer, scheduler, epochs, cfg):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.epochs = epochs

        self.reset()

    def reset(self):
        self.iter = 0
        self.loc_loss = 0
        self.conf_loss = 0

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.scheduler.step()

    def valid(self):
        pass

    def train_one_epoch(self, epoch):
        tqdm_loader = tqdm(self.train_loader)
        for b_idx, batch in tqdm_loader:
            loss, loss_l, loss_c = self.train_step(batch)
            self.iter += 1
            tqdm_loader.set_description(
                f'Epoch: {epoch}/{self.epochs} | Iter: {self.iter} | Loss: ' +
                f'total: {loss} - loc: {loss_l} - conf: {loss_c}'
            )

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images, targets = self.prepare_data(batch)
        out = self.model(images)
        loss_l, loss_c = self.loss(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        self.optimizer.step()
        self.loc_loss += loss_l.data[0]
        self.conf_loss += loss_c.data[0]
        return loss.item(), loss_l.item(), loss_c.item()

    def valid_step(self, batch):
        self.model.valid()
        with torch.no_grad():
            images, targets = self.prepare_data(batch)
        pass

    def prepare_data(self, batch):
        images, targets = batch
        images = images.to(self.device)
        targets = [Variable(anno.to(self.device), volatile=True) for anno in
                   targets]
        return images, targets
