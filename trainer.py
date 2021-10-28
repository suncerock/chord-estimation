"""
Trainer of the model
Referring from timm
"""
import argparse

import torch
import torch.nn as nn

import time

from dataloader import build_dataloader
from utils import *
from models.HarmonyTransformer import HarmonyTransformer
from configs.config_HarmonyTransformer import Config


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    # config, args = _parse_args()
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO: to build model from config
    model = HarmonyTransformer().to(device)
    train_dataloader, valid_dataloader = build_dataloader()
    optimizer = build_optimizer(model, config.optimizer_cfg)

    lr_scheduler, num_epoch = build_scheduler(optimizer, config.scheduler_cfg)
    start_epoch = 0
    if config.start_epoch is not None:
        start_epoch = config.start_epoch

    train_loss_fn = None
    valid_loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, num_epoch):
        print(epoch)
        train_one_epoch(epoch, model, train_dataloader, optimizer, train_loss_fn,
                        lr_scheduler=lr_scheduler, log_interval=config.log_interval)
        validate(model, valid_dataloader, valid_loss_fn)
        

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, lr_scheduler=None, log_interval=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        output, loss = model(batch)

        losses_m.update(loss.item(), output.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']

            print(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g}) '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s    '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)    '
                'LR: {lr:.3e} '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx,
                    len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=output.size(0) / batch_time_m.val,
                    rate_avg=output.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m
                )
            )


def validate(model, loader, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            output, loss = model(batch.to(device))

            # TODO: support MSRA, whether to use mir_eval package
            acc = (output.argmax(dim=-1) == batch[2]).float().mean()

            losses_m.update(loss.item(), output.size(0) * output.size(1))
            acc_m.update(acc.item(), output.size(0) * output.size(1))

            batch_time_m.update(time.time() - end)

    print(
        'Valid  '
        'Loss: {loss.val:#.4g} ({loss.avg:#.3g}) '
        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s    '
        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)    '
        'Accuracy: {acc.val:>7.4f} (acc.avg:>7.4f)'.format(
            loss=losses_m,
            batch_time=batch_time_m,
            rate=output.size(0) / batch_time_m.val,
            rate_avg=output.size(0) / batch_time_m.avg,
            acc=acc_m
            )
        )


if __name__ == '__main__':
    main()