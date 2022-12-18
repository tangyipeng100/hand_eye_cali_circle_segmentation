import torch
import numpy as np
import time
import pandas as pd
import os
import argparse
import open3d as o3d
from utils.Handeye_datasets import Handeye_datasets
import utils.config as con

def infer_vis():
    file_path = os.path.join(con.dataset_pre, con.vis_file_name)#sample path and loading
    model = torch.load(con.model_path)#trained model path and loading

    #load data
    sampled_profilesLabel = pd.read_csv(file_path.replace('\n', ''), sep=',', header=None)
    sampled_profilesLabel = np.array(sampled_profilesLabel)[0:con.points_n, :]

    #fill points
    if sampled_profilesLabel.shape[0] < con.points_n:
        sub_array = np.array([-199.0, -1, 0]) if sampled_profilesLabel.shape[1] == 3 else np.array([-199.0, -1])
        sub_array = np.repeat(sub_array.reshape(1, -1), con.points_n - sampled_profilesLabel.shape[0], axis=0)
        sampled_profilesLabel = np.vstack([sampled_profilesLabel, sub_array])

    sampled_profilesLabel = torch.tensor(sampled_profilesLabel, dtype=torch.float32).unsqueeze(0)
    un_profile_r = sampled_profilesLabel[:, :, 0:2]
    un_profile = un_profile_r[:, :, [0, 1]]

    #inference
    torch.cuda.synchronize()
    forward_start_time = time.time()

    un_profile = un_profile.cuda()
    centroid = torch.mean(un_profile, axis=1)

    un_profile = un_profile - centroid.unsqueeze(1)#these 2 steps consume a lot of time
    norm_unprofile = torch.max(torch.sqrt(torch.sum(un_profile ** 2, axis=2)), axis=1, keepdim=True)[0]
    un_profile = un_profile / norm_unprofile.unsqueeze(1)

    with torch.no_grad():
        model.eval()
        un_profile = un_profile.transpose(2, 1)
        model = model.cuda()
        out_l = model(un_profile)
        _, pred = torch.max(out_l, 1)


    torch.cuda.synchronize()
    forward_time_cost_100 = (time.time() - forward_start_time)
    print('Forward time:%.4f' % forward_time_cost_100)

    #plot and visualization
    points_p = un_profile_r.view(-1, 2).numpy()
    points_p = np.insert(points_p, 1, 0, axis=1)#add y axis to plot
    label_p = pred.cpu().view(-1, 1).numpy()
    color = np.zeros((points_p.shape[0], 3))

    for j in range(con.points_n):
        if label_p[j, 0] == 0:
            color[j, :] = [1, 1, 0]#[1, 1, 0]
        elif label_p[j, 0] == 1:
            color[j, :] = [0, 0, 1]
        elif label_p[j, 0] == 2:
            color[j, :] = [0, 1, 0]
        elif label_p[j, 0] == 3:
            color[j, :] = [1, 0, 0]#1,0,0
        elif label_p[j, 0] == 4:
            color[j, :] = [127 / 255, 0, 55 / 255]  # 0,0,1
        elif label_p[j, 0] == 5:
            color[j, :] = [205/255, 106/255, 0]#0,0,1

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_p)
    #pd.DataFrame(np.hstack([points_p, label_p])).to_csv('./test.csv', header=0, index=0, float_format='%.3f', sep=',')#save the visulization results
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries_with_editing([point_cloud])


def infer_results():
    num_classes = 2#segmentation classes number
    model = torch.load(con.model_path)  # trained model path and loading
    test_path = os.path.join(con.dataset_pre, con.test_file_name)

    seg_classes = {0: 'Outlier', 1: 'Circle points'}
    shape_ious = {'Outlier': [], 'Circle points': []}

    shape_ious_value = []
    total_correctv = 0
    total_points = 0
    total_seen_class = [0 for _ in range(len(seg_classes))]
    total_correct_class = [0 for _ in range(len(seg_classes))]


    test_dataset = Handeye_datasets(test_path, con.dataset_pre+'test_set', normal_method='Whole', num_classes=num_classes, dim=2)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=con.infer_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    with torch.no_grad():
        model = model.eval()
        for batch_i, (profile_imv, profile_labelv, _, _) in enumerate(test_dataloader):
            profile_imv = profile_imv.float().permute(0, 2, 1)
            profile_labelv = profile_labelv.long().squeeze(2)

            if torch.cuda.is_available():
                profile_imv = torch.autograd.Variable(profile_imv).cuda()
                profile_labelv = torch.autograd.Variable(profile_labelv).cuda()

            out_v = model(profile_imv)
            _, pred_v = torch.max(out_v, 1)
            pred_np = pred_v.long().cpu().numpy()
            label_np = profile_labelv.long().cpu().numpy()
            num_correctv = np.sum(pred_np == label_np)
            total_correctv += num_correctv
            total_points += pred_v.cpu().numpy().shape[0] * pred_v.cpu().numpy().shape[1]

            for j in range(len(seg_classes)):
                if np.sum((pred_np == j) | (label_np == j)) != 0:
                    total_correct_class[j] += np.sum((pred_np == j) & (label_np == j))
                    total_seen_class[j] += np.sum((pred_np == j) | (label_np == j))


    for j in range(len(seg_classes)):
        if total_seen_class[j] != 0:
            iou = total_correct_class[j] / float(total_seen_class[j])
            shape_ious_value.append(iou)
            shape_ious[seg_classes[j]] = iou
        else:
            shape_ious[seg_classes[j]] = 'None'

    for key, value in shape_ious.items():
        if type(value) != str:
            print('%s IOU:%.5f' % (key, value))
        else:
            print('%s IOU:%S' % (key, value))
    print('Mean IOU: %.5f' % np.mean(shape_ious_value))
    print('val_acc:%.5f' % (total_correctv / float(total_points)))


def batch_circle_segmentation():
    num_classes = 2

    if not os.path.exists(con.output_base_dir):
        os.mkdir(con.output_base_dir)
    model = torch.load(con.model_path)  # trained model path and loading
    val_path = os.path.join(con.dataset_pre, con.test_file_name)

    val_dataset = Handeye_datasets(val_path, con.dataset_pre+'test_set', normal_method='Whole', num_classes=num_classes, dim=2)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    with torch.no_grad():
        for batch_i, (profile_imv, profile_labelv, file_path, point_on) in enumerate(val_dataloader):
            profile_imv = profile_imv.float().permute(0, 2, 1)

            if torch.cuda.is_available():
                profile_imv = torch.autograd.Variable(profile_imv).cuda()

            out_v = model(profile_imv)
            _, pred_v = torch.max(out_v, 1)
            pred_np = pred_v.long().permute(1, 0).cpu().numpy()
            profile_st = pd.read_csv(os.path.join(con.dataset_pre + 'test_set', file_path[0][1:]), sep=',', header=None)
            profile_st = np.array(profile_st).astype('float')

            out_p_l = np.hstack([profile_st[:, 0:2], pred_np[:point_on, :]])
            save_name = os.path.basename(file_path[0])
            pd.DataFrame(out_p_l).to_csv(os.path.join(con.output_base_dir, save_name), header=0, index=0,
                                           float_format='%.3f', sep=',')
    print('Batch circle segmentation processing done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer_vis',
                        help='mode optional parameters: infer_vis, infer_results, batch_circle_segmentation,'
                             'infer_vis: show one sample inference result; infer_results: evaluate testset results; '
                             'batch_circle_segmentation: output result files of the testset ')
    opt = parser.parse_args()
    if opt.mode == 'infer_vis':
        infer_vis()
    elif opt.mode == 'infer_results':
        infer_results()
    elif opt.mode == 'batch_circle_segmentation':
        batch_circle_segmentation()



