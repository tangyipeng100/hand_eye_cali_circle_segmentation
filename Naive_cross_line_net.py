from load_plot import profile_load
import load_plot
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

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, 840]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, 840]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, 840]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 512, 840]
        out5 = F.relu(self.bn5(self.conv5(out4))) # [B, 1024, 840]

        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 1024)#[B, 1024]
        expand = out_max.view(-1, 1024, 1).repeat(1, 1, N)#[B, 1024, 840]
        concat_r = torch.cat([expand, out1, out2, out3, out4, out5], 1)##[B, 64+128+128+512+1024+1024=2880, 840]

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

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, 840]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, 840]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, 840]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 512, 840]

        out_max = torch.max(out4, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 512)#[B, 512]
        expand = out_max.view(-1, 512, 1).repeat(1, 1, N)#[B, 512, 840]
        concat_r = torch.cat([expand, out1, out2, out3, out4], 1)##[B, 64+128+128+512+512=1344, 840]

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

        out1 = F.relu(self.bn1(self.conv1(points)))#[B, 64, 840]
        out2 = F.relu(self.bn2(self.conv2(out1)))#[B, 128, 840]
        out3 = F.relu(self.bn3(self.conv3(out2)))#[B, 128, 840]
        out4 = F.relu(self.bn4(self.conv4(out3)))#[B, 256, 840]

        out_max = torch.max(out4, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 256)#[B, 512]
        expand = out_max.view(-1, 256, 1).repeat(1, 1, N)#[B, 512, 840]
        concat_r = torch.cat([expand, out1, out2, out3, out4], 1)##[B, 64+128+128+256+256=832, 840]

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
        concat_r = torch.cat([expand, out1, out2, out3], 1)##[B, 64+128+128+128=448, 840]

        net = F.relu(self.bns1(self.convs1(concat_r)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        return net

def main():
    # profile_im, profile_label = profile_load()
    # print('profile_im shape', profile_im.shape)
    # # fps_point = load_plot.farthest_point_sample(profile_im, 700)
    # # fps_point = fps_point.squeeze(0)
    # # print('fps_point shape', fps_point.shape)
    # # fps_list = fps_point.numpy().tolist()
    # # orig_list = [i for i in range(846)]
    # #differ = list(set(orig_list).difference(set(fps_list)))
    #
    # xyz, label = load_plot.uniform_resampling(profile_im, profile_label, 840)
    #
    # #plt.plot(profile_im[0, :, 0], profile_im[0, :, 2], c='b', marker='*', linewidth=0.1)
    # #plt.plot(xyz[1, :, 0], xyz[1, :, 2], c='r', marker='o', linewidth=0.1)
    #
    # plt.figure()
    #
    # plt.subplot(2, 1, 1)
    # plt.scatter(xyz[1, label[1, :, 0] == 1, 0], xyz[1, label[1, :, 0] == 1, 2], c='r', marker='*', linewidth=0.1)
    # plt.scatter(xyz[1, label[1, :, 0] == 2, 0], xyz[1, label[1, :, 0] == 2, 2], c='b', marker='*', linewidth=0.1)
    # plt.scatter(xyz[1, label[1, :, 0] == 3, 0], xyz[1, label[1, :, 0] == 3, 2], c='g', marker='*', linewidth=0.1)
    #
    # # plt.plot(profile_im[0, differ, 0], profile_im[0, differ, 2], c='r', marker='o', linewidth=0.1)
    # plt.subplot(2, 1, 2)
    # plt.scatter(profile_im[1, profile_label[1, :, 0] == 1, 0], profile_im[1, profile_label[1, :, 0] == 1, 2], c='r', marker='*', linewidth=0.1)
    # plt.scatter(profile_im[1, profile_label[1, :, 0] == 2, 0], profile_im[1, profile_label[1, :, 0] == 2, 2], c='b', marker='*', linewidth=0.1)
    # plt.scatter(profile_im[1, profile_label[1, :, 0] == 3, 0], profile_im[1, profile_label[1, :, 0] == 3, 2], c='g', marker='*', linewidth=0.1)
    # plt.show()

    # profile_im = profile_im.permute(0, 2, 1)
    #model = Naive_1d()
    #out = model(profile_im)
    #print('out shape', out.shape)
    model = _1024_pointnet(2, 2)
    xyz = torch.rand(5, 2, 1280)
    test_out =  (model(xyz))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')



if __name__ == '__main__':
    main()