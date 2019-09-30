#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_time = 2019/9/30 15:21:00
__author__ = qiuping1
__version__ = v_1.0.0
description:
change log: 
    2019/9/30 15:21:00: create file and edit code
"""
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim


def make_data():
    num_inputs = 2
    samples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.tensor(np.random.normal(0, 1, (samples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, labels.shape), dtype=torch.float)
    return features, labels


def get_data_iter(features, labels, batch_size):
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    return data_iter


def model(num_inputs):
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(num_inputs, 1))
    # for param in net.parameters():
    #     print(param)
    # 初始化参数
    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0.0)
    # for param in net.parameters():
    #     print(param)
    return net


def train():
    features, labels = make_data()
    batch_size = 10
    data_iter = get_data_iter(features, labels, batch_size)
    epochs = 5
    net = model(2)
    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    # 为不同子网络设置不同的学习率
    # optimizer =optim.SGD([
    #                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
    #                 {'params': net.subnet1.parameters()}, # lr=0.03
    #                 {'params': net.subnet2.parameters(), 'lr': 0.01}
    #             ], lr=0.03)
    # # 调整学习率
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
    for epoch in range(epochs):
        for x, y in data_iter:
            output = net(x)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))
    return net


if __name__ == '__main__':
    my_model = train()
    print(my_model, my_model[0].weight.data, my_model[0].bias.data)


