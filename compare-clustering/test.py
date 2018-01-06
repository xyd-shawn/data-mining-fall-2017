# -*- coding: utf-8 -*-

import os
import sys

import argparse
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models.cluster import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('-t', '--thresh', type=float, default=0.1, help='threshold of distance')
    parser.add_argument('-k', '--kernel', default='gaussian', help='kernel method')
    parser.add_argument('-m', '--metric', default='euclidean', help='distance metric')
    args = parser.parse_args()
    thresh, kernel, metric = args.thresh, args.kernel, args.metric
    data = np.loadtxt(config.input_file)
    X, labels = data[:, :-1], data[:, -1]
    nclusters = int(np.max(labels))
    print("origin classes: {}".format(nclusters))
    print(">>> start clustering")
    print(">>> DensityPeaks method")
    clf_1 = clusterDensityPeaks(thresh=thresh, kernel=kernel, metric=metric)
    nclusters_1, labels_1, idx_1 = clf_1.fit(X, save_file=config.decision_file)
    print("clusters number: {}".format(nclusters_1))
    print(">>> KMeans method")
    clf_2 = KMeans(n_clusters=nclusters)
    clf_2.fit(X)
    labels_2 = clf_2.labels_
    print(">>> DBSCAN method")
    clf_3 = DBSCAN(metric=metric)
    clf_3.fit(X)
    labels_3 = clf_3.labels_
    nclusters_3 = int(np.max(labels_3)) + 1
    print("clusters number: {}".format(nclusters_3))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    color = cm.rainbow(np.linspace(0, 1, nclusters))
    for i in range(nclusters):
        plt.scatter(X[labels==(i+1), 0], X[labels==(i+1), 1], color=color[i])
    plt.title("origin clusters")
    plt.subplot(2, 2, 2)
    color = cm.rainbow(np.linspace(0, 1, nclusters_1))
    for i in range(nclusters_1):
        plt.scatter(X[labels_1==i, 0], X[labels_1==i, 1], color=color[i])
    plt.title("DensityPeaks method")
    plt.subplot(2, 2, 3)
    color = cm.rainbow(np.linspace(0, 1, nclusters))
    for i in range(nclusters):
        plt.scatter(X[labels_2==i, 0], X[labels_2==i, 1], color=color[i])
    plt.title("KMeans method")
    plt.subplot(2, 2, 4)
    color = cm.rainbow(np.linspace(0, 1, nclusters_3 + 1))
    for i in range(nclusters_3 + 1):
        plt.scatter(X[labels_3==(i-1), 0], X[labels_3==(i-1), 1], color=color[i])
    plt.title("DBSCAN method")
    plt.savefig(config.out_file)
    plt.show()
