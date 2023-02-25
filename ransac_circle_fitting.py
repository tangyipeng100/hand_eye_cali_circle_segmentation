#!/usr/bin/env python
# coding=utf-8

import math
import numpy as np
import pandas as pd
import os
import utils.config as con


points_x = []
points_y = []

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    Refrence: http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
    Pseudocode: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
    Input:
        data: Fitting points
        model: Fitting models
        n: the least number of points need to generate the model
        k: the maximum iteration number
        t: inlier points judgement threshold
        d: the least number of points need to fit
    Output:
        bestfit: the best fitting solution
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if len(also_inliers > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class CircleLeastSquareModel:

    def __init__(self):
        pass

    def fit(self, data):
        A = []
        B = []
        for d in data:
            A.append([-d[0], -d[1], -1])
            B.append(d[0] ** 2 + d[1] ** 2)
        A_matrix = np.array(A)
        B_matrix = np.array(B)
        C_matrix = A_matrix.T.dot(A_matrix)
        result = np.linalg.inv(C_matrix).dot(A_matrix.T.dot(B_matrix))
        self.a = result[0]
        self.b = result[1]
        self.c = result[2]
        return self


    def get_error(self, data, model):
        err_per_point = []
        for d in data:
            B = d[0] ** 2 + d[1] ** 2
            B_fit = model.a * d[0] + model.b * d[1] + model.c
            err_per_point.append((B + B_fit) ** 2)  # sum squared error per row
        return np.array(err_per_point)

def sort_regu(file_name):
    sf = file_name.split('_')
    return int(sf[0]) if sf[0]!='m' else int(sf[1])


if __name__ == "__main__":
    average_center = []
    temp_p = []

    fitting_m = 'RANSAC'
    for root, _, files in os.walk(con.output_base_dir):
        files.sort(key=sort_regu)
        print(files)
        for i, f in enumerate(files):

            test_p = np.array(pd.read_csv(os.path.join(root, f), header=None, sep=','))
            test_p = test_p[test_p[:, 2] == 1, 0:2]
            points_x = test_p[:, 0].tolist()
            points_y = test_p[:, 1].tolist()

            model = CircleLeastSquareModel()
            data = np.vstack([points_x, points_y]).T
            result = model.fit(data)
            x0 = result.a * -0.5
            y0 = result.b * -0.5
            r = 0.5 * math.sqrt(result.a ** 2 + result.b ** 2 - 4 * result.c)

            # run RANSAC algorithm
            ransac_fit, ransac_data = ransac(data, model, 20, 1000, 2, 10, debug=False, return_all=True)
            x1 = ransac_fit.a * -0.5
            y1 = ransac_fit.b * -0.5
            r_1 = 0.5 * math.sqrt(ransac_fit.a ** 2 + ransac_fit.b ** 2 - 4 * ransac_fit.c)
            points_x_array = np.array(points_x)
            points_y_array = np.array(points_y)

            w_minus = True if os.path.basename(f)[0:2] == 'm_' else False
            if fitting_m == 'RANSAC':
                sc_x = x1
                sc_y = -math.sqrt(con.ref_ball_r ** 2 - r_1 ** 2) if w_minus else math.sqrt(con.ref_ball_r ** 2 - r_1 ** 2)
                sc_z = y1
            else:
                sc_x = x0
                sc_y = -math.sqrt(con.ref_ball_r ** 2 - r ** 2) if w_minus else math.sqrt(con.ref_ball_r ** 2 - r ** 2)
                sc_z = y0

            if i%5 != 4:
                if fitting_m == 'RANSAC':
                    temp_p.append([sc_x, sc_y, sc_z, r_1])
                else:
                    temp_p.append([sc_x, sc_y, sc_z, r])
            else:
                temp_p = np.array(temp_p)
                average_center.append(np.mean(temp_p, 0))
                temp_p = []
    average_center = np.array(average_center)
    pd.DataFrame(average_center[:, 0:4]).to_csv(os.path.join(con.output_save_dir, 'average_center.csv'), header=0, index=0,
                                 float_format='%.6f', sep=',')
    print(average_center)