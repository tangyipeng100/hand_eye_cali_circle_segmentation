import torch
import numpy as np
import pandas as pd
from csv import reader
import os
import natsort
import open3d as o3d
import matplotlib.pyplot as plt

# Point label defination:
#    0 -- Mould points
#    1 -- Prepreg points
#    2 -- Gap points
#    3 -- Defect points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
    # 其中B为BatchSize的个数
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices初始化为0~(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 直到采样点达到npoint，否则进行如下迭代：
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest
        # 取出该中心点centroid的坐标
        debug_cen = xyz[batch_indices, farthest, :]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小，
        # 其相当于记录着某个Batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return centroids

def uniform_resampling(xyz, label, n_resample):
    """

    :param xyz: pointcloud data, [B, N, 3] tensor
    :param label: label data, [B, N, 1] tensor
    :param n_resample: number of resamples
    :return: resample_xyz: resampled pointcloud data, [B, n_resample, 3] tensor
    :return: resample_label: resampled pointcloud label, [B, n_resample, 1] tensor
    """
    if xyz.shape[1] != label.shape[1]:
        print('Error dimension')
        return

    B = xyz.shape[0]
    resample_xyz = torch.zeros(B, n_resample, 3)
    resample_label = torch.zeros(B, n_resample, 1)

    for i in range(B):
        xyz_b = xyz[i, :, :].numpy()#[N, 3]
        label_b = label[i, :, :].numpy()#[N, 1]
        idx = np.argwhere(np.all(xyz_b[:, :] == -1, axis=1))
        xyz_b = np.delete(xyz_b, idx, 0)
        label_b = np.delete(label_b, idx, 0)

        resample_xyz[i, :, 1] = xyz[i, 0, 1]
        x_st = xyz_b[0, 0]
        x_en = xyz_b[-1, 0]
        x_resample = np.linspace(x_st, x_en, n_resample)
        n_findx = 0#find x number in xyz_b[n_findx, 0]
        for j in range(n_resample):
            for k in range(n_findx, xyz_b.shape[0]-1):
                if x_resample[j] >= xyz_b[k, 0] and x_resample[j] <= xyz_b[k+1, 0]:
                    resample_xyz[i, j, 0] = x_resample[j]
                    #calculate the weight to assign values
                    w1 = (x_resample[j] - xyz_b[k, 0])/(xyz_b[k+1, 0] - xyz_b[k, 0])
                    w2 = (xyz_b[k+1, 0] - x_resample[j])/(xyz_b[k+1, 0] - xyz_b[k, 0])
                    resample_xyz[i, j, 2] = w2 * xyz_b[k, 2]+w1 * xyz_b[k+1, 2]
                    p1 = np.array([xyz_b[k, 0], xyz_b[k, 2]])
                    p2 = np.array([resample_xyz[i, j, 0], resample_xyz[i, j, 2]])
                    p3 = np.array([xyz_b[k+1, 0], xyz_b[k+1, 2]])
                    d12 = np.sqrt(np.sum((p1-p2)**2, -1))
                    d23 = np.sqrt(np.sum((p2-p3)**2, -1))
                    if d12 > d23:
                        resample_label[i, j, 0] = int(label_b[k+1, 0])
                    else:
                        resample_label[i, j, 0] = int(label_b[k, 0])


                    #resample_label[i, j, 0] = int(round(wd2 * label_b[k, 0] + wd1 * label_b[k+1, 0]))
                    n_findx = k
    return resample_xyz, resample_label


def uniform_list_resampling(xyz, label, n_resample):
    """

    :param xyz: pointcloud data, B*(**n*3) list
    :param label: label data, B*(**n) list
    :param n_resample: number of resamples
    :return: resample_xyz: resampled pointcloud data, [B, n_resample, 3] tensor
    :return: resample_label: resampled pointcloud label, [B, n_resample, 1] tensor
    """
    if len(xyz) != len(label):
        print('Error dimension')
        return

    B = len(xyz)
    resample_xyz = torch.zeros(B, n_resample, 3)
    resample_label = torch.zeros(B, n_resample, 1)

    for i in range(B):
        xyz_b = xyz[i]#[N, 3]
        label_b = label[i].reshape(-1, 1)#[N, 1]

        resample_xyz[i, :, 1] = xyz_b[0, 1]
        x_st = xyz_b[0, 0]
        x_en = xyz_b[-1, 0]
        x_resample = np.linspace(x_st, x_en, n_resample)
        n_findx = 0#find x number in xyz_b[n_findx, 0]
        for j in range(n_resample):
            for k in range(n_findx, xyz_b.shape[0]-1):
                if x_resample[j] >= xyz_b[k, 0] and x_resample[j] <= xyz_b[k+1, 0]:
                    resample_xyz[i, j, 0] = x_resample[j]
                    #calculate the weight to assign values
                    w1 = (x_resample[j] - xyz_b[k, 0])/(xyz_b[k+1, 0] - xyz_b[k, 0])
                    w2 = (xyz_b[k+1, 0] - x_resample[j])/(xyz_b[k+1, 0] - xyz_b[k, 0])
                    resample_xyz[i, j, 2] = w2 * xyz_b[k, 2]+w1 * xyz_b[k+1, 2]
                    p1 = np.array([xyz_b[k, 0], xyz_b[k, 2]])
                    p2 = np.array([resample_xyz[i, j, 0], resample_xyz[i, j, 2]])
                    p3 = np.array([xyz_b[k+1, 0], xyz_b[k+1, 2]])
                    d12 = np.sqrt(np.sum((p1-p2)**2, -1))
                    d23 = np.sqrt(np.sum((p2-p3)**2, -1))
                    if d12 > d23:
                        resample_label[i, j, 0] = int(label_b[k+1, 0])
                    else:
                        resample_label[i, j, 0] = int(label_b[k, 0])


                    #resample_label[i, j, 0] = int(round(wd2 * label_b[k, 0] + wd1 * label_b[k+1, 0]))
                    n_findx = k
    return resample_xyz, resample_label

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    #pc = pc / m
    return pc

def profile_load():
    """Load profile and label from saved csv files

    :return:
        proflile_im: 2x840x2
        profile_label: 2x840x1
    """
    profile_1 = []
    filename = '/home/tangyipeng/文档/学习资料/data/anno_57_final/ex_anno_57_8.csv'
    with open(filename, 'rt') as anno_profile:
        readers = reader(anno_profile, delimiter=',')
        x = list(readers)
        profile_1 = np.array(x).astype('float')

    # profile_p1 = profile_1[:, 0:2]
    # profile_p1 = pc_normalize(profile_p1)

    n_point1 = profile_1.shape[0]

    profile_2 = []
    filename = '/home/tangyipeng/文档/学习资料/data/anno_57_final/ex_anno_57_85.csv'
    with open(filename, 'rt') as anno_profile:
        readers = reader(anno_profile, delimiter=',')
        x = list(readers)
        profile_2 = np.array(x).astype('float')
    n_point2 = profile_2.shape[0]


    # plt.scatter(profile_2[profile_2[:,2]==1, 0], profile_2[profile_2[:,2]==1, 1], c = 'r', marker='*', linewidth=0.1)
    # plt.scatter(profile_2[profile_2[:,2]==2, 0], profile_2[profile_2[:,2]==2, 1], c = 'b', marker='*', linewidth=0.1)
    # plt.scatter(profile_2[profile_2[:,2]==3, 0], profile_2[profile_2[:,2]==3, 1], c = 'g', marker='*', linewidth=0.1)
    # plt.show()

    proflile_im = -torch.ones(2, 900, 4)
    profile_label = -torch.ones(2, 900, 1)
    proflile_im[0, 0:n_point1, 0:3] = torch.tensor(profile_1[:, 0:3])
    proflile_im[1, 0:n_point2, 0:3] = torch.tensor(profile_2[:, 0:3])
    profile_label[0, 0:n_point1, 0] = torch.tensor(profile_1[:, 3])
    profile_label[1, 0:n_point2, 0] = torch.tensor(profile_2[:, 3])
    #print(profile_label.shape)
    return proflile_im, profile_label

def main():
    file_dircoll = ['/home/tangyipeng/文档/学习资料/data/anno_57_final', \
                    '/home/tangyipeng/文档/学习资料/data/anno_58_final', \
                    '/home/tangyipeng/文档/学习资料/data/anno_59_final', \
                    '/home/tangyipeng/文档/学习资料/data/anno_60_final', \
                    ]
    file_path = file_dircoll[3]
    for root, dirs, files in os.walk(file_path):
        files = natsort.natsorted(files)
        n_p = len(files)
        sampled_profiles = np.zeros((n_p*840, 4))
        for i, file in enumerate(files):
            if i > 30:
                break
            files[i] = os.path.join(root, file)
            with open(files[i], 'rt') as anno_profile:
                readers = reader(anno_profile, delimiter=',')
                x = list(readers)
                profile_1 = np.array(x).astype('float')
                profile_point = torch.tensor(profile_1[:, 0:3]).unsqueeze(0)
                profile_label = torch.tensor(profile_1[:, 3]).unsqueeze(0).unsqueeze(2)
                xyz, label = uniform_resampling(profile_point, profile_label, 840)
                sampled_profiles[i*840:(i+1)*840, 0:3] = xyz.squeeze(0).numpy()
                sampled_profiles[i*840:(i+1)*840, 3] = label.squeeze(0).squeeze(1).numpy()
                print(i)


    color = np.zeros((sampled_profiles.shape[0], 3))
    for j in range(i*840):
        if sampled_profiles[j, 3] == 1:
            color[j, :] = [1, 0, 0]
        elif sampled_profiles[j, 3] == 2:
            color[j, :] = [0, 1, 0]
        else:
            color[j, :] = [0, 0, 1]
    idx = np.argwhere(np.all(color[:, :] == 0, axis=1))
    color = np.delete(color, idx, 0)
    sampled_profiles = np.delete(sampled_profiles, idx, 0)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(sampled_profiles[:, 0:3])
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([point_cloud])

    #pd.DataFrame(sampled_profiles).to_csv(os.path.join(file_path, 'export_wholepoint.csv'), header=0, index=0, sep=',')
    print('done')

if __name__ == '__main__':
    main()