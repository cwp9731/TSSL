# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :model_utils.py
# @Time      :2024/4/12 11:01
# @Author    :Chen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(C_in, C_out, 7, 3),
            nn.BatchNorm1d(C_out),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layer(x)


class BN_Conv1d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object, dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv1d, self).__init__()
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm1d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """

    def __init__(self, in_channels, out_channels, strides):
        super(BasicBlock, self).__init__()
        self.conv1 = BN_Conv1d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv1d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)


        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.short_cut(x)
        return F.relu(out)



class CNN(nn.Module):
    """ CNN for Self-Supervision """

    def __init__(self, embedding_dim=256, device=''):
        super(CNN, self).__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.device = device
        self.embedding_dim = embedding_dim
        self.f = nn.Sequential(
            Conv(1, 4),
            Conv(4, 16),
            Conv(16, 64)
        )
        self.f1 = nn.Linear(8 * 64, self.embedding_dim)
        self.g = nn.Sequential(nn.Linear(embedding_dim, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, embedding_dim, bias=True))

    def forward(self, x):
        batch_size = x.shape[0]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        outs = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]
            h = self.f(h)
            h = h.view(h.size(0), -1)
            feature = self.f1(h)
            out = self.g(feature)
            latent_embeddings[:, :, n] = F.normalize(feature, dim=-1)
            outs[:, :, n] = F.normalize(out, dim=-1)
        return latent_embeddings, outs


class AlexNet(nn.Module):
    def __init__(self, embedding_dim=256, device=''):
        super(AlexNet, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        frist_channel = 4
        self.f = torch.nn.Sequential(
            torch.nn.Conv1d(1, frist_channel, kernel_size=11, stride=4, padding=2),
            torch.nn.BatchNorm1d(frist_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),

            torch.nn.Conv1d(frist_channel, 3*frist_channel, kernel_size=7, padding=2),
            torch.nn.BatchNorm1d(3*frist_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),

            torch.nn.Conv1d(3*frist_channel, 9*frist_channel, kernel_size=5, padding=1),
            torch.nn.BatchNorm1d(9*frist_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(9*frist_channel,  6*frist_channel, kernel_size=5, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d( 6*frist_channel),
            torch.nn.Conv1d(6*frist_channel,  6*frist_channel, kernel_size=5, padding=1),
            torch.nn.BatchNorm1d(6*frist_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.f1 = nn.Linear(59 * 6 * frist_channel, self.embedding_dim)
        self.g = nn.Sequential(nn.Linear(embedding_dim, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, embedding_dim, bias=True))


    def forward(self, x):
        batch_size = x.shape[0]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        outs = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]
            h = self.f(h)
            h = h.view(h.size(0), -1)
            feature = self.f1(h)
            out = self.g(feature)
            latent_embeddings[:, :, n] = F.normalize(feature, dim=-1)
            outs[:, :, n] = F.normalize(out, dim=-1)
        return latent_embeddings, outs


class VGG(nn.Module):
    def __init__(self, embedding_dim=256, device=''):
        super(VGG, self).__init__()
        self.device = device
        frist_kera_size = 4
        self.embedding_dim = embedding_dim
        self.f = torch.nn.Sequential(

            torch.nn.Conv1d(1, frist_kera_size, kernel_size=3, stride=4,  padding=2),
            torch.nn.BatchNorm1d(frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(frist_kera_size, frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(frist_kera_size, 4*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(4*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4*frist_kera_size, 4*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(4*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(4*frist_kera_size,  8*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d( 8*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d( 8*frist_kera_size,  8*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d( 8*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d( 8*frist_kera_size,  8*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d( 8*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d( 8*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),

            torch.nn.Conv1d(16*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16*frist_kera_size, 16*frist_kera_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16*frist_kera_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
        )
        self.f1 = nn.Linear(32 * 16 * frist_kera_size, self.embedding_dim)
        self.g = nn.Sequential(nn.Linear(embedding_dim, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, embedding_dim, bias=True))


    def forward(self, x):
        batch_size = x.shape[0]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        outs = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]
            h = self.f(h)
            h = h.view(h.size(0), -1)
            feature = self.f1(h)
            out = self.g(feature)
            latent_embeddings[:, :, n] = F.normalize(feature, dim=-1)
            outs[:, :, n] = F.normalize(out, dim=-1)
        return latent_embeddings, outs




class ResNet(nn.Module):
    def __init__(self, embedding_dim=256, device=''):
        super(ResNet, self).__init__()
        channels = 4
        self.channels = channels  # out channels from the first convolutional layer
        patches = self.channels * 72
        self.block = BasicBlock

        self.embedding_dim = embedding_dim

        self.device = device
        # encoder
        self.f = nn.Sequential(
            nn.Conv1d(1, self.channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.MaxPool1d(3, 2, 1),
            self._make_conv_x(channels=channels, blocks=2, strides=1, index=2),
            self._make_conv_x(channels=channels * 4, blocks=2, strides=2, index=3),
            self._make_conv_x(channels=channels * 8, blocks=2, strides=2, index=4),
            self._make_conv_x(channels=channels * 16, blocks=2, strides=2, index=5),
        )
        self.f1 = nn.Linear(64*channels*16, embedding_dim)
        # projection head
        self.g = nn.Sequential(nn.Linear(embedding_dim, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, embedding_dim, bias=True))


    def _make_conv_x(self, channels, blocks, strides, index):
        list_strides = [strides] + [1] * (blocks - 1)
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels
        return conv_x

    def forward(self, x):
        batch_size = x.shape[0]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        outs = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]
            h = self.f(h)
            h = h.view(h.size(0), -1)
            feature = self.f1(h)
            out = self.g(feature)
            latent_embeddings[:, :, n] = F.normalize(feature, dim=-1)
            outs[:, :, n] = F.normalize(out, dim=-1)
        return latent_embeddings, outs






class SeNet(nn.Module):
    def __init__(self, frist_model, noutputs, embedding_dim=256):
        super(SeNet, self).__init__()
        self.frist_model = frist_model
        self.linear = nn.Linear(embedding_dim*24, noutputs)
        self.embedding_dim = embedding_dim*24

    def forward(self, x):
        x, _ = self.frist_model(x)
        h = x.view(-1, self.embedding_dim)
        output = self.linear(h)
        return output


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)



if __name__ == '__main__':
    model_list = ['CNN', 'AlexNet', 'VGG', 'ResNet']
    for model_name in model_list:
        if model_name == model_list[0]:
            model = CNN(128, 'cpu')
        elif model_name == model_list[1]:
            model = AlexNet(128, 'cpu')
        elif model_name == model_list[2]:
            model = VGG(128, 'cpu')
        elif model_name == model_list[3]:
            model = ResNet(128, 'cpu')
        input = torch.randn(size=(16, 1, 2048, 24))
        latent_embeddings = model(input)
        print(model_name)
        getModelSize(model)
