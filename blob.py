# -*- coding: UTF-8 -*-
#
# File Name    :    blob.py
# Author       :    doujm
# Mail         :    doujm@jiedaibao.com
# Create Time  :    2017-08-19 22:14:47
############################################### 

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.datasets.samples_generator import make_blobs

n_samples = 500
centers = [[1, 1], [-1, -2], [4, -1]]
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2,
        random_state=0)
print X.shape
print y

#y[:n_samples // 2] = 0
#y[n_samples // 2:] = 1
#print y
#print X[:, 0]
#print X[:, 1]
#print y.shape[0]

y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
sample_weight = np.random.RandomState(42).rand(y.shape[0])

X_train, X_test, y_train, y_test, sw_train, sw_test = \
        train_test_split(X, y, sample_weight, test_size=0.4, random_state=42)
print X_train.shape
print X_test.shape
print sw_train.shape


print colors
plt.figure()

for this_y, color in zip(y_unique, colors):
    print y_train == this_y
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color,
        alpha=0.5, edgecolor='k', label="Class %s" % this_y)

#plt.scatter(X[:, 0], X[:, 1], color = colors, cmap = plt.cm.spectral, alpha=0.5, edgecolor='k')
plt.show()
