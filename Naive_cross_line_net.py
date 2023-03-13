import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class _1024_pointnet(nn.Module):
    def __init__(self, cat_num=4, dim=2):
        super(_1024_pointnet, self).__init__()
        self.cat_num = cat_num
        self._dim = dim
        self.conv1 = torch.nn.Conv1d(self._dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.convs0 = torch.nn.Conv1d(2880, 1024, 1)
        self.convs1 = torch.nn.Conv1d(1024, 512, 1)
        self.convs2 = torch.nn.Conv1d(512, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, cat_num, 1)
        self.bns0 = nn.BatchNorm1d(1024)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, points):
        B, D, N = points.size()

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, N]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, N]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, N]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 512, N]
        out5 = F.relu(self.bn5(self.conv5(out4))) # [B, 1024, N]

        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 1024)#[B, 1024]
        expand = out_max.view(-1, 1024, 1).repeat(1, 1, N)#[B, 1024, N]
        concat_r = torch.cat([expand, out1, out2, out3, out4, out5], 1)##[B, 64+128+128+512+1024+1024=2880, N]

        net = F.relu(self.bns0(self.convs0(concat_r)))
        net = F.relu(self.bns1(self.convs1(net)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

class _512_pointnet(nn.Module):
    def __init__(self, cat_num=4, dim=2):
        super(_512_pointnet, self).__init__()
        self.cat_num = cat_num
        self._dim = dim
        self.conv1 = torch.nn.Conv1d(self._dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)

        self.convs1 = torch.nn.Conv1d(1344, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, cat_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, points):
        B, D, N = points.size()

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, N]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, N]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, N]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 512, N]

        out_max = torch.max(out4, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 512)#[B, 512]
        expand = out_max.view(-1, 512, 1).repeat(1, 1, N)#[B, 512, 840]
        concat_r = torch.cat([expand, out1, out2, out3, out4], 1)##[B, 64+128+128+512+512=1344, N]

        net = F.relu(self.bns1(self.convs1(concat_r)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

class _256_pointnet(nn.Module):
    def __init__(self, cat_num=4, dim=2):
        super(_256_pointnet, self).__init__()
        self.cat_num = cat_num
        self._dim = dim

        self.conv1 = torch.nn.Conv1d(self._dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.convs1 = torch.nn.Conv1d(832, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, cat_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, points):
        B, D, N = points.size()

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, N]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, N]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, N]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 256, N]

        out_max = torch.max(out4, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 256)#[B, 512]
        expand = out_max.view(-1, 256, 1).repeat(1, 1, N)#[B, 512, N]
        concat_r = torch.cat([expand, out1, out2, out3, out4], 1)##[B, 64+128+128+256+256=832, N]

        net = F.relu(self.bns1(self.convs1(concat_r)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

class _128_pointnet(nn.Module):
    def __init__(self, cat_num=4, dim=2):
        super(_128_pointnet, self).__init__()
        self.cat_num = cat_num
        self._dim = dim

        self.conv1 = torch.nn.Conv1d(self._dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.convs1 = torch.nn.Conv1d(448, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, cat_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, points):
        B, D, N = points.size()

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, 840]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, 840]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, 840]

        out_max = torch.max(out3, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 128)#[B, 512]
        expand = out_max.view(-1, 128, 1).repeat(1, 1, N)#[B, 512, 840]
        concat_r = torch.cat([expand, out1, out2, out3], 1)##[B, 64+128+128+128=448, N]

        net = F.relu(self.bns1(self.convs1(concat_r)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

class _64_pointnet(nn.Module):
    def __init__(self, cat_num=4, dim=2):
        super(_64_pointnet, self).__init__()
        self.cat_num = cat_num
        self._dim = dim

        self.conv1 = torch.nn.Conv1d(self._dim, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.convs1 = torch.nn.Conv1d(224, 128, 1)
        self.convs3 = torch.nn.Conv1d(128, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, cat_num, 1)
        self.bns1 = nn.BatchNorm1d(128)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, points):
        B, D, N = points.size()

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 32, 840]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 64, 840]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 64, 840]

        out_max = torch.max(out3, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 64)#[B, 512]
        expand = out_max.view(-1, 64, 1).repeat(1, 1, N)#[B, 512, 840]
        concat_r = torch.cat([expand, out1, out2, out3], 1)##[B, 32+64+64+64=224, N]

        net = F.relu(self.bns1(self.convs1(concat_r)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

def main():

    # model = _1024_pointnet(2, 2)
    model = _64_pointnet(2, 2)
    xyz = torch.rand(5, 2, 1280)
    test_out = (model(xyz))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')



if __name__ == '__main__':
    main()