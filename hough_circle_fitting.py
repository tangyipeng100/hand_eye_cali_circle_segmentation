import numpy as np
import pandas as pd
import time

#fitting file loading
file_path_n = './data/circle_fitting_files/60_noise.csv'
test_p = np.array(pd.read_csv(file_path_n, header=None, sep=','))
point = test_p[:, [0, 1]]

#subdivision radius r
r_min = 14
r_max = 16
r_size = 0.1
r_cir = np.arange(r_min, r_max, r_size)

#Subdivision center parameters a, b
a_cir = np.arange(np.min(point[:, 0]), np.max(point[:, 0]), 0.1)
b_cir = np.arange(np.min(point[:, 1]), np.max(point[:, 1]), 0.1)


def DetectCircleHough(points):
    points = points[200:700, :]
    A=np.zeros((len(a_cir), len(b_cir), len(r_cir)+1))#define the accumulation unit
    #calculate the accumulation unit
    for p in points:
        x=p[0]
        y=p[1]
        for a in range(len(a_cir)):
            for b in range(len(b_cir)):
                r = np.sqrt((a_cir[a]-x)*(a_cir[a]-x)+(b_cir[b]-y)*(b_cir[b]-y))
                if r>r_min and r<r_max:
                    A[a, b, round((r-r_min)/r_size)] +=1
    return A


if __name__=='__main__':

    forward_start_time = time.time()
    A = DetectCircleHough(point)
    forward_time_cost_100 = (time.time() - forward_start_time)

    print('Forward time:%.4f' % forward_time_cost_100)
    #Print the index corresponding to the maximum value of the accumulation unit
    out_index = np.where(A == A.max())
    print(np.where(A == A.max()))
    print('x: %.3f, y: %.3f, r: %.3f' % (a_cir[out_index[0][0]].item(), b_cir[out_index[1][0]].item(), r_cir[out_index[2][0]].item()))

