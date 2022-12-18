# -*- coding: utf-8 -*-
"""
# @file name  : my_lr_schedule.py
# @author     : https://github.com/TingsongYu
# @date       : 2021-03-06 10:08:00
# @brief      : 学习率下降策略工具
"""
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.models import resnet18
from math import pi, cos
from torch.optim.optimizer import Optimizer


class CosineWarmupLr(object):
    """Cosine lr decay function with warmup.
    Lr warmup is proposed by `
        Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by `
        Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        batches (int): batches of one epoch.
        max_epochs (int): max_epochs to train.
        base_lr (float): init lr.
        final_lr (float): minimum(final) lr.
        warmup_epochs (int): warmup max_epochs before cosine decay.
        warmup_init_lr (float): warmup starting lr.
        last_iter (int): init iteration.
    Attributes:
        niters (int): number of iterations of all max_epochs.
        warmup_iters (int): number of iterations of all warmup max_epochs.
    """
    def __init__(self, optimizer, batches, max_epochs, base_lr, final_lr=0,
                 warmup_epochs=0, warmup_init_lr=0, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.baselr = base_lr
        self.learning_rate = base_lr
        self.niters = max_epochs * batches
        self.targetlr = final_lr
        self.warmup_iters = batches * warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        self.last_iter = last_iter
        self.step()

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            self.learning_rate = self.warmup_init_lr + \
                (self.baselr - self.warmup_init_lr) * self.last_iter / self.warmup_iters
        else:
            self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
                (1 + cos(pi * (self.last_iter - self.warmup_iters) /
                         (self.niters - self.warmup_iters))) / 2

    def step(self, iteration=None):
        """Update status of lr.
        Args:
            iteration(int, optional): now training iteration of all max_epochs.
                Normally need not to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate


if __name__ == '__main__':
    
    warmup_epochs = 10  #
    max_epoch = 500  # 共有120个epoch，则用于cosine rate的一共有100个epoch
    lr_init = 1e-4
    lr_final = 1e-6
    lr_warmup_init = 0.

    iter_per_epoch = 75

    model = resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    scheduler = CosineWarmupLr(optimizer, batches=iter_per_epoch, max_epochs=max_epoch, base_lr=1e-3,
                               final_lr=lr_final, warmup_epochs=warmup_epochs, warmup_init_lr=lr_warmup_init)
    index = 0
    x = []
    y = []
    for epoch in range(max_epoch):
        for iter in range(iter_per_epoch):
            lr_c = optimizer.param_groups[0]['lr']
            y.append(lr_c)
            #print(lr_c)
            scheduler.step()
            x.append(index)
            index += 1
        print(epoch, lr_c)
    plt.figure(figsize=(10, 8))
    plt.xlabel('epoch')
    plt.ylabel('cosine rate')
    plt.plot(x, y, color='r', linewidth=2.0, label='cosine rate')
    plt.legend(loc='best')
    plt.show()

