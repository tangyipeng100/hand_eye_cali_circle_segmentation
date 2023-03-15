from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
import numpy as np
from csv import reader
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random

# import warnings
# warnings.simplefilter('error')

class Handeye_datasets(Dataset):
    def __init__(self, file_path, dataset_pre, normal_method='Whole', num_classes=4, dim=2, point_n=1280):
        self._dim = dim
        self.num_classes = num_classes
        self.normal_method = normal_method
        self.point_n = point_n
        self.dataset_pre = dataset_pre

        with open(file_path, 'r') as file:
            self.profile_files = file.readlines()
        label_st = np.zeros(self.num_classes)
        for profile_name in self.profile_files:
            profile_st = pd.read_csv(os.path.join(dataset_pre, profile_name.replace('\n', '')[1:]), sep=',', header=None)
            profile_st = np.array(profile_st).astype('float')

            label_temp = profile_st[:, 2]
            if label_temp.shape[0] < self.point_n:
                label_temp = np.hstack([label_temp, np.zeros(self.point_n-label_temp.shape[0])])
            tmp, _ = np.histogram(label_temp, range(self.num_classes+1))
            label_st += tmp
        label_st = label_st.astype(np.float32)
        label_st = label_st / np.sum(label_st)
        self.labelweights = np.power(np.amax(label_st) / label_st, 1 / 3.0)#计算和归一权重



    def __getitem__(self, index):
        file_path = self.profile_files[index % len(self.profile_files)].rstrip()
        profile_f = pd.read_csv(os.path.join(self.dataset_pre, file_path.replace('\n', '')[1:]), sep=',', header=None)
        profile_f = np.array(profile_f).astype('float')
        point_n = profile_f.shape[0]
        if profile_f.shape[0] < self.point_n:
            sub_array = np.array([-199.0, -1, 0])
            sub_array = np.repeat(sub_array.reshape(1, -1), self.point_n-profile_f.shape[0], axis=0)
            profile_f = np.vstack([profile_f, sub_array])

        if self._dim == 2:
            profile_im = profile_f[:, [0, 1]]#[n, 2]
        elif self._dim == 3:
            profile_im = profile_f[:, [0, 1, 2]]#[n, 3]


        if self.normal_method == 'Whole':
            profile_im = self.pc_normalize(profile_im)  # [n, 3]
        elif self.normal_method == 'Min_max':
            profile_im = self.pc_normalize_MinMax(profile_im)#[n, 3]
        profile_label = np.expand_dims(profile_f[:, 2], 1)#[n, 1]
        #profile_im, profile_label = self.uniform_resampling(profile_im, profile_label, 840)#[840, 2],torch:[840, 1]
        profile_label = torch.tensor(profile_label)

        return profile_im, profile_label, file_path, point_n#[N, 3], [N]

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / (m+0.0001)
        return pc

    def pc_rotation(self, data_or, argu_angle):
        '''
        Point cloud rotation
        :param data_or: Nx4, point clouds with labels
        :param argu_angle: float, rotation angle
        :return: data_or: NX3, Rotated point clouds
        '''

        data_label = data_or[:, 3]
        data_or = data_or[:, 0:3]
        fit_num_label = 1  # using which label to fit the rotation plane
        if sum(data_label == 1) < 20:
            fit_num_label = np.argmax( np.bincount(np.array(data_label, dtype=int)) )#find the label that most points belongs to
        label1_pc = data_or[data_label == fit_num_label, :]
        data_fitting = label1_pc - np.mean(label1_pc, 0)

        X_i = data_fitting[:, 0]
        Y_i = data_fitting[:, 1]
        Z_i = data_fitting[:, 2]

        M_svd = np.array([[np.sum(X_i * X_i), np.sum(X_i * Y_i), np.sum(X_i * Z_i)],
                          [np.sum(X_i * Y_i), np.sum(Y_i * Y_i), np.sum(Y_i * Z_i)],
                          [np.sum(X_i * Z_i), np.sum(Y_i * Z_i), np.sum(Z_i * Z_i)]
                          ])
        _, _, VT = np.linalg.svd(M_svd)
        v_r = VT[2, :]
        data_pr = np.hstack([data_or, np.ones([data_or.shape[0], 1])])
        p_rotated = np.matmul(self.rotation_matrix(argu_angle, np.mean(label1_pc, 0), v_r), data_pr.transpose())
        data_ro = p_rotated.transpose()[:, 0:3]

        return data_ro

    def rotation_matrix(self, r_angle, p_m, v_r):
        '''
        Rotation matrix generation.l
        :param r_angle: float
        :param p_m: numpy, [4,], middle point
        :param v_r: numpy, [3,],
        :return: M_T, numpy, [4, 4]
        '''
        M_rotation = np.zeros([4, 4])
        u, v, w = v_r[0], v_r[1], v_r[2]
        Cos = math.cos(r_angle)
        Sin = math.sin(r_angle)
        M_t1 = np.array([[1, 0, 0, -p_m[0]], [0, 1, 0, -p_m[1]], [0, 0, 1, -p_m[2]], [0, 0, 0, 1]])
        M_t2 = np.array([[1, 0, 0, p_m[0]], [0, 1, 0, p_m[1]], [0, 0, 1, p_m[2]], [0, 0, 0, 1]])

        M_rotation[0, 0] = u * u * (1 - Cos) + Cos  # u * u + (v * v + w * w) * Cos
        M_rotation[0, 1] = u * v * (1 - Cos) - w * Sin  # u * v * (1 - Cos) - w * Sin
        M_rotation[0, 2] = u * w * (1 - Cos) + v * Sin
        M_rotation[0, 3] = 0  # (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - Cos) + (b * w - c * v) * Sin
        M_rotation[1, 0] = u * v * (1 - Cos) + w * Sin
        M_rotation[1, 1] = v * v * (1 - Cos) + Cos  # v * v + (u * u + w * w) * Cos
        M_rotation[1, 2] = v * w * (1 - Cos) - u * Sin
        M_rotation[1, 3] = 0  # (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - Cos) + (c * u - a * w) * Sin
        M_rotation[2, 0] = u * w * (1 - Cos) - v * Sin
        M_rotation[2, 1] = v * w * (1 - Cos) + u * Sin
        M_rotation[2, 2] = w * w * (1 - Cos) + Cos  # w * w_ (u * u + v * v) * Cos
        M_rotation[2, 3] = 0  # (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - Cos) + (a * v - b * u) * Sin
        M_rotation[3, 0] = 0
        M_rotation[3, 1] = 0
        M_rotation[3, 2] = 0
        M_rotation[3, 3] = 1

        M_T = np.matmul(M_t2, np.matmul(M_rotation, M_t1))
        return M_T

    def pc_normalize_MinMax(self, pc):
        pc[:, 0] = (pc[:, 0] - np.mean(pc[:, 0])) / (np.max(pc[:, 0]) - np.min(pc[:, 0]) + 0.1)
        pc[:, 1] = (pc[:, 1] - np.mean(pc[:, 1])) / (np.max(pc[:, 1]) - np.min(pc[:, 1]) + 0.1)
        pc[:, 2] = (pc[:, 2] - np.mean(pc[:, 2])) / (np.max(pc[:, 2]) - np.min(pc[:, 2]) + 0.1)
        return pc

    def uniform_resampling(self, xyz, label, n_resample):
        """

        :param xyz: pointcloud data, [N, 2] array
        :param label: label data, [N, 1] array
        :param n_resample: number of resamples
        :return: resample_xyz: resampled pointcloud data, [n_resample, 3] array
        :return: resample_label: resampled pointcloud label, [n_resample, 1] array
        """

        resample_xyz = np.zeros((n_resample, 2))
        resample_label = torch.zeros((n_resample, 1))

        #xyz_b = xyz[i, :, :].numpy()
        #label_b = label[i, :, :].numpy()
        #idx = np.argwhere(np.all(xyz_b[:, :] == -1, axis=1))
        #xyz_b = np.delete(xyz_b, idx, 0)
        #label_b = np.delete(label_b, idx, 0)

        x_st = xyz[0, 0]
        x_en = xyz[-1, 0]
        x_resample = np.linspace(x_st, x_en, n_resample)
        n_findx = 0  # find x number in xyz_b[n_findx, 0]
        for j in range(n_resample):
            for k in range(n_findx, xyz.shape[0] - 1):
                if x_resample[j] >= xyz[k, 0] and x_resample[j] <= xyz[k + 1, 0]:
                    resample_xyz[j, 0] = x_resample[j]
                    # calculate the weight to assign values
                    w1 = (x_resample[j] - xyz[k, 0]) / (xyz[k + 1, 0] - xyz[k, 0])
                    w2 = (xyz[k + 1, 0] - x_resample[j]) / (xyz[k + 1, 0] - xyz[k, 0])
                    resample_xyz[j, 1] = w2 * xyz[k, 1] + w1 * xyz[k + 1, 1]
                    p1 = np.array([xyz[k, 0], xyz[k, 1]])
                    p2 = np.array([resample_xyz[j, 0], resample_xyz[j, 1]])
                    p3 = np.array([xyz[k + 1, 0], xyz[k + 1, 1]])
                    d12 = np.sqrt(np.sum((p1 - p2) ** 2, -1))
                    d23 = np.sqrt(np.sum((p2 - p3) ** 2, -1))
                    if d12 > d23:
                        resample_label[j, 0] = int(label[k + 1, 0])
                    else:
                        resample_label[j, 0] = int(label[k, 0])

                    # resample_label[i, j, 0] = int(round(wd2 * label_b[k, 0] + wd1 * label_b[k+1, 0]))
                    n_findx = k
        return resample_xyz, resample_label

    def __len__(self):
        return len(self.profile_files)

# def main():
#     dataset_path = '../data/Standard_sphere_seg_dataset_v1/train_file.txt'
#
#     train_dataset = Handeye_datasets(dataset_path, normal_method='Whole', num_classes=2, dim=2)
#     print(train_dataset.labelweights)
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=5,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=False,
#     )
#
#
#     #sem & ins dataloader
#     for batch_i, (profile_im, profile_label, _, _) in enumerate(train_dataloader):
#         print(profile_im.shape)
#
#
#
# if __name__ == '__main__':
#     main()

