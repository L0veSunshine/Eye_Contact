import torch
import time
import os
from torch import optim
from torch.utils.data import DataLoader
from datasets.landmark import Landmark
from models.loss_fn import WingLoss
from models.net import SlimNet
from utils.util import *

# configs
lr_decay_every_epoch = [1, 25, 55, 75, 150]
lr_value_every_epoch = [0.00001, 0.0001, 0.00005, 0.00001, 0.000001]
weight_decay_factor = 5.e-4
l2_regularization = weight_decay_factor
input_size = (160, 160)
batch_size = 256
wing_loss_fn = WingLoss()
train_epoch = 151


class Metrics:
    def __init__(self):
        self.landmark_loss = 0
        self.counter = 0

    def update(self, landmark_loss):
        self.landmark_loss += landmark_loss.item()
        self.counter += 1

    def summary(self):
        return self.landmark_loss / self.counter


def dynamic_lr(epoch):
    if epoch < lr_decay_every_epoch[0]:
        return lr_value_every_epoch[0]
    elif lr_decay_every_epoch[0] <= epoch < lr_decay_every_epoch[1]:
        return lr_value_every_epoch[1]
    elif lr_decay_every_epoch[1] <= epoch < lr_decay_every_epoch[2]:
        return lr_value_every_epoch[2]
    elif lr_decay_every_epoch[2] <= epoch < lr_decay_every_epoch[3]:
        return lr_value_every_epoch[3]
    elif lr_decay_every_epoch[3] <= epoch < lr_decay_every_epoch[4]:
        return lr_value_every_epoch[4]
    else:
        return 0.000001


def train(epoch):
    model.train()
    metrics = Metrics()
    total_samples = 0
    start = time.time()
    print("==================================Training Epoch: {} =================================".format(epoch))
    print("当前学习率:{}".format(list(optim.param_groups)[0]['lr']))
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        optim.zero_grad()
        preds = model(imgs)
        loss = 3 * wing_loss_fn(preds, labels)
        metrics.update(loss)
        loss.backward()
        optim.step()
        total_samples += len(imgs)
        progress = total_samples / len(train_dataset)
        etc = (time.time() - start) / total_samples * (len(train_dataset) - total_samples)
        rewrite("Loss: {:.4f} Progress: {:.2f}% Etc: {:.0f}s".format(loss.item(), progress * 100, etc))
    next_line()
    avg_loss = metrics.summary()
    print("Train Avg Loss -- {:.4f}".format(avg_loss))


def test(epoch):
    if epoch % 5 == 0 and epoch > 0:
        model.eval()
        metrics = Metrics()
        start = time.time()
        total_samples = 0
        print("==================================Eval Epoch: {} =================================".format(epoch))
        for i, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                preds = model(imgs)
                loss = wing_loss_fn(preds, labels)
            metrics.update(loss)
            total_samples += len(imgs)
            progress = total_samples / len(val_dataset)
            etc = (time.time() - start) / total_samples * (len(val_dataset) - total_samples)
            rewrite("Loss -- Total: {:.4f} Progress: {:.4f}% Etc: {:.0f}/s".
                    format(loss.item(), progress * 100, etc))
        next_line()
        avg_loss = metrics.summary()
        print("Eval Avg Loss: {:.4f}".format(avg_loss))
        torch.save(model.state_dict(), open("weight/slim128_epoch_{}_{:.4f}.pth".format(epoch, avg_loss), "wb"))
    else:
        torch.save(model.state_dict(), open("weight/slim128_epoch_{}.pth".format(epoch), "wb"))


if __name__ == '__main__':
    checkpoint = None
    train_dataset = Landmark("train.json", input_size, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = Landmark("val.json", input_size, True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SlimNet().cuda()
    model.train()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        start_epoch = int(checkpoint.split("_")[-2]) + 1
    else:
        start_epoch = 0

    optim = optim.Adam(model.parameters(), lr=lr_value_every_epoch[0], weight_decay=5e-4)
    for epoch in range(start_epoch, train_epoch):
        for param_group in optim.param_groups:
            param_group['lr'] = dynamic_lr(epoch)
        train(epoch)
        test(epoch)
