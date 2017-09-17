# -*- coding: UTF-8 -*-
#
# File Name    :    bsm.py
# Author       :    doujm
# Mail         :    doujm@jiedaibao.com
# Create Time  :    2017-09-17 13:14:02
############################################### 

import os
import sys
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs

def print_statistics(a1, a2):
    """
    print selected statistics.
    Parameters
    ==========
    a1, a2 : ndarray objects results object from simulation
    """
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print "%14s %14s %14s" % ("statistic", "data set 1", "data set 2")
    print 45 * "-"
    print "%14s %14.3f %14.3f" % ("size", sta1[0], sta2[0])
    print "%14s %14.3f %14.3f" % ("min", sta1[1][0], sta2[1][0])
    print "%14s %14.3f %14.3f" % ("max", sta1[1][1], sta2[1][1])
    print "%14s %14.3f %14.3f" % ("mean", sta1[2], sta2[2])
    print "%14s %14.3f %14.3f" % ("std", np.sqrt(sta1[3]), np.sqrt(sta2[3]))
    print "%14s %14.3f %14.3f" % ("skew", sta1[4], sta2[4])
    print "%14s %14.3f %14.3f" % ("kurtosis", sta1[5], sta2[5])

#S0 = 100
#r = 0.05
#sigma = 0.25
#T = 2.0

sigma = 0.25
T = 2.0
S0 = 10
r = 0.05

I = 10000
M = 80
dt = T / M
S = np.zeros((M + 1, I))
S[0] = S0

for t in range(1, M + 1):
    xline = random.standard_normal(I)
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + 
            sigma * np.sqrt(dt) * xline)

print_statistics(S[:, 0], S[:, 1])
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
#print St1[:100]
#plt.hist(S[-1], bins = 50)
#plt.plot(St1)
#plt.hist(xline, bins = 5000)
#plt.plot(xline)
plt.xlabel("index label")
plt.ylabel('frequency')
plt.grid(True)
#plt.show()



