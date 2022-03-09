import torch


def my_collate(batch):
    images, targets, hw = [], [], []
    for item in batch:
        images.append(item[0].unsqueeze(0))
        targets.append(torch.FloatTensor(item[1]))
        hw.append(item[2])

    return torch.cat(images, dim=0), targets, hw
