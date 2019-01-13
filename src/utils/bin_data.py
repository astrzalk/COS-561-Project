#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import gamma as Gamma
from numpy.random import normal as Normal

def bin_single_router_pair(traffic_mat, router_pair, which_transform=1, max_bin_size=100, seed=561):
    i, j = router_pair
    xs = traffic_mat[:, i, j]
    np.random.seed(seed)
    ys = flow_transform(xs, which_transform)
    num_pts = len(xs)
    binned_ys = np.reshape(ys, (int(num_pts / max_bin_size), max_bin_size))
    time_to_predict_inds =\
    [i for i in range(max_bin_size, num_pts, max_bin_size)] + [num_pts - 1]
    xs_to_predict = [x for ind, x in enumerate(xs) if ind in time_to_predict_inds]
    return (binned_ys, np.array(xs_to_predict))

def flow_transform(x, which_transform):
    """
    input
    -----
    x : numpy array, representing univariate time series data.
    which_transform : Takes values 1 or 2. If it is 1, we transform x
                      according to M1 in
                      'Predicting Future Trafﬁc using Hidden Markov Models'.
                      If 2, we transform x according to M2.
    output
    ------
    y : numpy array, transformation of x by either M1, or M2.

    """
    shape = 1 # assume shape of gammas is 1?
    if which_transform == 1:
        y = 0.01 * (x + 0.05 * (Gamma(shape) + Normal()))
    else:
        y = 0.01 * (np.power(x, 0.1) + 0.25 * (Gamma(shape) + Normal()))
    return y

if __name__ == "__main__":
    data_path = "../data/traffic_mats.npy"
    traffic_mats = np.load(data_path)[0:48000, :, :]
    router_pair = (3, 4)
    list_of_ys = []
    for _ in range(10):
        ys, xs = bin_single_router_pair(traffic_mats, router_pair)
        list_of_ys.append(ys)
    # Make sure random seed works
    assert(np.linalg.norm(list_of_ys[0] - list_of_ys[7]) == 0)
    assert(ys.shape == (480, 100))
    assert(xs.shape == (480,))

