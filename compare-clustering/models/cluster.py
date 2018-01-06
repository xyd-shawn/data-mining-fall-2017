# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

from utils import *


class clusterDensityPeaks(object):

    def __init__(self, **kwargs):
        '''
        Options:
        kernel: string, 'cutoff' or 'gaussian'
        metric: string, distance metric
        thresh: float, threshold of distance
        '''
        self.kernel = kwargs.get('kernel', 'cutoff')
        self.metric = kwargs.get('metric', 'euclidean')
        self.thresh = kwargs.get('thresh', 0.1)

    def fit(self, X, **kwargs):
        dist = pairwise_distances(X, metric=self.metric)
        dc = self.decision_dc(dist)
        rho, idx1 = self.compute_kernel(dist, dc)
        delta, nn = self.compute_delta(dist, rho, idx1)
        gamma = (rho / np.max(rho)) * (delta / np.max(delta))
        self.plot_decision_graph(rho, delta, gamma, **kwargs)
        nclusters = int(input('>>> input the number of clusters\n>>> '))
        idx2 = sort_index(gamma)
        cc = -np.ones_like(rho, dtype='int')
        for i in range(nclusters):
            cc[idx2[i]] = i
        for i in range(len(idx1)):
            if cc[idx1[i]] == -1:
                cc[idx1[i]] = cc[nn[idx1[i]]]
        return nclusters, cc, idx2

    def decision_dc(self, dist):
        n = dist.shape[0]
        arr = np.array([dist[i][j] for i in range(n) for j in range(i + 1, n)])
        dc = np.percentile(arr, int(self.thresh * 100))
        return dc

    def compute_kernel(self, dist, dc):
        if self.kernel == 'gaussian':
            rho = np.sum(np.exp(-(dist ** 2) / dc), axis=1) - 1
        else:
            rho = np.sum(dist < dc, axis=1) - 1
        idx1 = sort_index(rho)
        return rho, idx1

    def compute_delta(self, dist, rho, idx1):
        delta = np.zeros_like(rho)
        nn = -np.ones_like(rho, dtype='int')
        delta[idx1[0]] = np.max(dist[idx1[0], :])
        max_dist = dist.max() + 1
        for i in range(1, len(rho)):
            delta[idx1[i]] = max_dist
            for j in range(i):
                if dist[idx1[i], idx1[j]] < delta[idx1[i]]:
                    delta[idx1[i]] = dist[idx1[i], idx1[j]]
                    nn[idx1[i]] = idx1[j]
        return delta, nn

    def plot_decision_graph(self, rho, delta, gamma, **kwargs):
        imag_show = kwargs.get('imag_show', True)
        save_file = kwargs.get('save_file', None)
        fig, axe = plt.subplots(1, 2)
        axe[0].scatter(rho, delta, color='b', marker='o')
        axe[1].scatter(range(1, 1 + len(gamma)), sorted(gamma, reverse=True), color='b', marker='o')
        if imag_show:
            plt.show()
        if save_file:
            plt.savefig(save_file)

    def compute_halo(self, rho, nclusters, dc, cc, dist):
        h = np.zeros_like(rho, dtype='int')
        rhob = np.zeros(nclusters)
        for i in range(len(rho) - 1):
            for j in range(i + 1, len(rho)):
                if (cc[i] != cc[j]) and (dist[i, j] < dc):
                    temp = (rho[i] + rho[j]) / 2
                    if temp > rhob[cc[i]]:
                        rhob[cc[i]] = temp
                    if temp > rhob[cc[j]]:
                        rhob[cc[j]] = temp
        for i in range(len(rho)):
            if rho[i] < rhob[cc[i]]:
                h[i] = 1
        return h
