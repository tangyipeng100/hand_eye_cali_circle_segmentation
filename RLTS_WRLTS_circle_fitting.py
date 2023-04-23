from ransac_circle_fitting import CircleLeastSquareModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import sys
from matplotlib.patches import Ellipse, Circle
import circle_fit as cf
import argparse
import os
import time

'''
 Code implementation of paper: Robust statistical approaches for circle fitting in laser scanning 
 three-dimensional point cloud data, Nurunnabi, Sadahiro, 10.1016/j.patcog.2018.04.010
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algebratic_m', type=str, default='LSQ', help='Least square method or HyperLSQ')
    parser.add_argument('--robust_m', type=str, default='WLTS', help='RLTS or WRLTS')
    parser.add_argument('--h', type=float, default=10, help='Select points number for the second fitting')
    parser.add_argument('--h0', type=float, default=3, help='Select points number for the first fitting')
    parser.add_argument("--pr", type=float, default=0.999, help='Monte Carlo type probabilistic parameter')
    parser.add_argument("--eta", type=float, default=0.5, help='Monte Carlo type probabilistic parameter')
    parser.add_argument("--fitting_file", type=str, default='./data/circle_fitting_files/3_noise.csv', help='Monte Carlo type probabilistic parameter')

    opt = parser.parse_args()
    return opt

def get_error_RLTS(data, xc, yc, f_r):
    err_s_per_point = []
    for d in data:
        err = math.sqrt((d[0] - xc) ** 2 + (d[1] - yc) ** 2) - f_r
        err_s_per_point.append(err ** 2)
    return np.array(err_s_per_point)

if __name__ == "__main__":
    # Hyper parameters settings
    opt = parse_args()
    algebratic_m = opt.algebratic_m# LSQ or HyperLSQ
    robust_m = opt.robust_m # WRLTS or RLTS

    h, h0, pr, eta = opt.h, opt.h0, opt.pr, opt.eta # Iteration number
    In = math.log(1-pr)/math.log(1-(1-eta)**h0)
    #In = 1

    file_path = opt.fitting_file

    test_p = np.array(pd.read_csv(file_path, header=None, sep=','))

    test_p = test_p[:, 0:2]
    points_x = test_p[:, 0].tolist()
    points_y = test_p[:, 1].tolist()
    #
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    plt.plot(points_x, points_y, "ro", label="Data points")
    plt.axis('equal')
    min_sse = sys.maxsize
    fit_r = []

    model = CircleLeastSquareModel()  # Instantiation of a class: generating a known model using least squares.
    data = np.vstack([points_x, points_y]).T
    st_time = time.time()
    for _ in range(int(In)):
        # first fitting
        random_r_in = np.random.choice(data.shape[0], size=h0, replace=False)
        fit_d = data[random_r_in, :]
        try:
            if algebratic_m == 'LSQ':
                result = model.fit(fit_d)
                a = 1
            else:
                x0, y0, r0, _ = cf.hyperLSQ(fit_d)
            #a = 1
        except Exception as e:
            print(e)

        if algebratic_m == 'LSQ':
            x0 = result.a * -0.5
            y0 = result.b * -0.5
            err_n0 = model.get_error(data, result)
            in_d = err_n0.argsort()[:h]
        else:
            err_n0 = get_error_RLTS(data, x0, y0, r0)
            in_d = err_n0.argsort()[:h]

        # fitting again
        if algebratic_m == 'LSQ':
            result_ag = model.fit(data[in_d, :])
            x1 = result_ag.a * -0.5
            y1 = result_ag.b * -0.5
            r1 = 0.5 * math.sqrt(result.a ** 2 + result.b ** 2 - 4 * result.c)
            err_n1 = model.get_error(data, result_ag)
        else:
            x1, y1, r1, _ = cf.hyperLSQ(data[in_d, :])
            err_n1 = get_error_RLTS(data, x1, y1, r1)

        err_w = np.zeros_like(err_n1)
        for i in range(err_w.shape[0]):
            if err_n1[i] < 1:
                err_w[i] = (1-err_n1[i])**2
            else:
                err_w[i] = 0
        if robust_m == 'RLTS':
            err_w = np.ones_like(err_n1)
        err_n1 = err_n1 * err_w

        if np.sum(err_n1) < min_sse:
            min_sse = np.sum(err_n1)
            fit_r = [x1, y1, r1]

    print('fitting time: ', time.time()-st_time)
    # plot
    circle2 = Circle(xy=(fit_r[0], fit_r[1]), radius=fit_r[2], fill=False, label=robust_m + ' fitting', color='b')
    plt.gcf().gca().add_artist(circle2)
    ax.add_patch(circle2)
    plt.xlabel('X mm', fontfamily='Times New Roman')
    plt.ylabel('Y mm', fontfamily='Times New Roman')
    _title = robust_m + ' ' + os.path.basename(opt.fitting_file).split('.')[0] + 'data'
    plt.title(_title)
    plt.grid()
    plt.legend(loc='upper right')
    plt.axis('equal')

    print("circle x is %f, y is %f, r is %f" % (fit_r[0], fit_r[1], fit_r[2]))
    #plt.savefig(_title + '.png', dpi=300, bbox_inches='tight')
    plt.show()


