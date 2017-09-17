# -*- coding: UTF-8 -*-
#
# File Name    :    relateMap.py
# Author       :    doujm
# Mail         :    doujm@jiedaibao.com
# Create Time  :    2017-08-18 21:07:36
############################################### 

import os
import sys
import hashlib
import numpy as np
import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
from sklearn import cluster, covariance, manifold

def retry(f, n_attempts = 3):
    def wrapper(*args, **kwargs):
        for i in range(n_attempts):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if i == n_attempts - 1:
                    raise
    return wrapper

def quotes_historical_google(symbol, date1, date2):
    """ Get the historical data from google finance.
    Parameters
    __________
    symbol : str
        Ticker sysmbol to query for, for example ``"DELL"``
    date1 : datetime.datetime
        Start date
    date2 : datetime.datetime
        End date.
    Returns
    _______
    X : array
        The columns are ``date`` --dattime, ``open``, ``high``,
        ``low``, ``close`` and ``volume`` of type float
    """
    params = urlencode({
        'q'  : symbol,
        'startdate' : date1.strftime('%b %d, %Y'),
        'enddae' : date2.strftime('%b %d, %Y'),
        'output' : 'csv'
    })
    url  = 'http://www.google.com/finance/historical?' + params
    md5 = hashlib.md5(url.encode('UTF-8')).hexdigest()
    path = "/Users/tianyi/project/stock/data/"
    md5File = path + md5 + ".txt"

    #response = urlopen(url)
    if (os.path.isfile(md5File)):
        response = open(md5File).read()
    else:
        print(md5File, url)
        response = urlopen(url).read()
        #print response
        fd = open(md5File, 'w')
        fd.write(response)
        #fd.write(response.encode("UTF-8"))
        fd.close()
    dtype = {
        'names' : ['date', 'open', 'high', 'low', 'close', 'volume'],
        'formats' : ['object', 'f4', 'f4', 'f4', 'f4', 'f4']
    }
    converters = {0: lambda s: datetime.strptime(s.decode(), '%d-%b-%y')}
    res = np.genfromtxt(StringIO(response), delimiter = ',', skip_header = 1,
            dtype = dtype, converters = converters, 
            missing_values = '-' , filling_values = -1)
    return  res

d1 = datetime(2003, 1, 1)
d2 = datetime(2008, 1, 1)

symbol_dict = {
    'TOT': 'total'
}


symbol_dict = {
    'TOT': 'Total',
    #'XOM': 'Exxon',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'VLO': 'Valero Energy',
    #'MSFT': 'Microsoft',
    'IBM': 'IBM',
    #'TWX': 'Time Warner',
    #'CMCSA': 'Comcast',
    #'CVC': 'Cablevision',
    #'YHOO': 'Yahoo',
    #'DELL': 'Dell',
    'HPQ': 'HP',
    #'AMZN': 'Amazon',
    #'TM': 'Toyota',
    #'CAJ': 'Canon',
    #'SNE': 'Sony',
    'F': 'Ford',
    #'HMC': 'Honda',
    'NAV': 'Navistar',
    'NOC': 'Northrop Grumman',
    #'BA': 'Boeing',
    'KO': 'Coca Cola',
    #'MMM': '3M',
    'MCD': 'McDonald\'s',
    'PEP': 'Pepsi',
    #'K': 'Kellogg',
    'UN': 'Unilever',
    #'MAR': 'Marriott',
    'PG': 'Procter Gamble',
    'CL': 'Colgate-Palmolive',
    'GE': 'Genera = Electrics',
    'WFC': 'Wells Fargo',
    'JPM': 'JPMorgan Chase',
    #'AIG': 'AIG',
    'AXP': 'American express',
    #'BAC': 'Bank of America',
    'GS': 'Goldman Sachs',
    #'AAPL': 'Apple',
    'SAP': 'SAP',
    #'CSCO': 'Cisco',
    #'TXN': 'Texas Instruments',
    #'XRX': 'Xerox',
    'WMT': 'Wal-Mart',
    'HD': 'Home Depot',
    #'GSK': 'GlaxoSmithKline',
    'PFE': 'Pfizer',
    'SNY': 'Sanofi-Aventis',
    'NVS': 'Novartis',
    'KMB': 'Kimberly-Clark',
    'R': 'Ryder',
    'GD': 'General Dynamics',
    'RTN': 'Raytheon',
    #'CVS': 'CVS',
    'CAT': 'Caterpillar',
    #'DD': 'DuPont de Nemours'
}
symbols, names = np.array(list(symbol_dict.items())).T

#print len(names)
#print len(symbol_dict.items())
#for idx in range(len(names)):
#    print idx, names[idx]
#print "____________________________"
quotes = [
    retry(quotes_historical_google)(symbol, d1, d2) for symbol in symbols
]

#closeArr = [q['close'] for q in quotes]
#print type(closeArr)
#print type(closeArr[0])
closeArr = []
cnt = 0
for q in quotes:
    print cnt, names[cnt], len(q)
    if len(q) < 3683:
        print cnt, names[cnt], len(q)
        continue;
    closeArr.insert(cnt, q["close"])
    cnt += 1
close_prices = np.vstack(closeArr)
openArr = []
cnt = 0
for q in quotes:
    #print len(q)
    if len(q) < 3683:
        continue;
    openArr.insert(cnt, q["open"])
    cnt += 1
open_prices = np.vstack(openArr)



#print(len(open_prices))
#print(open_prices)
variatition = close_prices - open_prices
#print variatition
edge_model = covariance.GraphLassoCV()
X = variatition.copy().T

#求每一个的标准方差
X /= X.std(axis=0)
#print X
#exit()
edge_model.fit(X)
#print edge_model.covariance_
print edge_model.covariance_.shape
print edge_model.covariance_[:4, :4]
#exit()
#print X.shape
#print edge_model.covariance_.shape
_, labels = cluster.affinity_propagation(edge_model.covariance_)
#print(labels)
#print edge_model.covariance_
n_labels = labels.max()

shape = edge_model.covariance_.shape

maxValue = 0.0
for x in xrange(0, shape[0]):
    for y in xrange(0, shape[1]):
        if x != y and edge_model.covariance_[x, y] > maxValue:
            maxValue = edge_model.covariance_[x, y]
            print maxValue, names[x], " : ", names[y]

#print maxValue
#exit()
for i in range(n_labels + 1):
    isPrint = False
    #print i
    #print labels == i
    print ("Cluster %i: %s" % ((i + 1), ','.join(names[labels == i])))
    #if i == 8:
    #    isPrint = True
    group = labels == i
    for idx in xrange(len(group)):
        #print group[idx]
        if group[idx] == False:
            continue
        if isPrint:
            print " ", names[idx], ":"
        for idy in xrange(idx + 1, len(group)):
            if group[idy] == False:
                continue
            #print idy, 
            if isPrint:
                print "Varience :", names[idy], ":", round(edge_model.covariance_[idx, idy], 2), "\t",
        if isPrint:
            print ""
            #print "_____________", ":", edge_model.covariance_[idx, :]
#exit()

#exit()

#LLE 算法，用于降纬, 使用KNN表示, 用于绘制容易理解的图
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, n_neighbors=6
    #n_components=2, eigen_solver='dense', n_neighbors=6
)
#print X.T
embedding = node_position_model.fit_transform(X.T).T
print X.T.shape
#print embedding
print embedding.shape
#exit()
plt.figure(1, facecolor = 'w', figsize = (10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
#plt.scatter(embedding[0], embedding[1], c = labels, cmap=plt.cm.spectral)
#plt.scatter(embedding[0], embedding[1], s = 100 * d ** 2, c = labels, cmap=plt.cm.spectral)
#plt.show()
start_idx, end_idx = np.where(non_zero)
segments = [[embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments, zorder = 0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dx[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size = 10, 
            horizontalalignment = horizontalalignment, 
            verticalalignment = verticalalignment, 
            bbox = dict(facecolor = 'w', 
                edgecolor=plt.cm.spectral(label / float(n_labels)),
                alpha=.6
            )
    )
plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
    embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
    embedding[1].max() + .03 * embedding[1].ptp())
print "ended"
plt.show()
