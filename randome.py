# -*- coding: UTF-8 -*-
#
# File Name    :    random.py
# Author       :    doujm
# Mail         :    doujm@jiedaibao.com
# Create Time  :    2017-09-17 11:29:46
############################################### 

import os
import sys
from numpy import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sample_size = 500

rn1 = random.standard_normal(sample_size)
rn2 = random.normal(100, 20, sample_size)
rn3 = random.chisquare(df = 0.5, size = sample_size)

#rn1 = random.rand(sample_size, 3)
#rn2 = random.randint(0, 10, sample_size)
#rn3 = random.sample(size = sample_size)
#a = [0, 25, 50, 75, 100]
#rn4 = random.choice(a, size = sample_size)

################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (7, 7))
ax1.hist(rn1, bins = 25, stacked = True)
ax1.set_title('rand')
ax1.set_ylabel("frequency")
ax1.grid(True)

ax2.hist(rn2, bins = 25)
ax2.set_title("randint")
ax2.grid(True)

ax3.hist(rn3, bins = 25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax3.grid(True)

ax4.hist(rn4, bins = 25)
ax4.set_title("choice")
ax4.grid(True)

plt.show()
