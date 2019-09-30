#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_time = 2019/9/30 14:39:00
__author__ = qiuping1
__version__ = v_1.0.0
description:
change log: 
    2019/9/30 14:39:00: create file and edit code
"""
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


def make_data():
    """
    生成数据集
    :return:
    """
    num_inputs = 2
    samples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(samples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    return features, labels


def get_data_iter(features, labels, batch):
    """
    获取数据
    :param features:
    :param labels:
    :param batch:
    :return:
    """
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch)
    return data_iter


def model():
    """
    定义模型
    :return:
    """
    net = nn.Sequential()  # 串联各连接层的容器
    net.add(nn.Dense(1))  # 添加一个全连接层
    net.initialize(init.Normal(sigma=0.01))  # 初始化模型参数
    return net


def train():
    features, labels = make_data()
    batch_size = 10
    data_iter = get_data_iter(features, labels, batch_size)
    epochs = 5
    net = model()
    loss = gloss.L2Loss()  # 定义损失函数，平方损失
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})  # 定义优化方法
    for epoch in range(epochs):
        for x, y in data_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
    return net


if __name__ == '__main__':
    my_model = train()
    print(my_model, my_model[0].weight.data(), my_model[0].bias.data())









