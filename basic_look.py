
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic(u'javascript', u'', u"IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from __future__ import print_function, division
from os.path import join, exists, dirname
import os
import re
import errno
from functools import partial
from glob import glob
import datetime as dt
from dateutil import rrule
from datetime import datetime, timedelta
import calendar
from collections import Counter
from operator import methodcaller as mc

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# import requests
# from bs4 import BeautifulSoup
import xlrd


pd.options.display.notebook_repr_html = False
pd.options.display.width = 140
get_ipython().magic(u'matplotlib inline')
sns.set_palette(sns.color_palette('colorblind', 5))


# In[ ]:

def drop_exc(df, cs):
    if not hasattr(cs, '__iter__'):
        cs = [cs]
    cs = map(str, cs)
    other_nums = [c for c in df if c.isdigit() and c not in cs]
    return df[[c for c in df if c not in other_nums]]

# drop_ex = partial(drop_exc, df=df0)
def just_nums(df):
    return [c for c in df if c.isdigit()]

def rotate(deg=90):
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=deg)
#     plt.plot(x, delay)

def read(site_loc, sheet=0):
    fn = join('data', site_loc, 'all_{}.msg'.format(sheet))
    exfile = glob(join('data', site_loc, '*.xlsx'))[0]
    
    xl_workbook = xlrd.open_workbook(exfile)
    sheet_names = xl_workbook.sheet_names()
    del xl_workbook

    print('{} => {}'.format(sheet_names, sheet_names[sheet]))
    with open(join('data', site_loc, 'description.txt')) as f:
        print(f.read())
        
        df = pd.read_msgpack(fn)
    print('Nulls: {} / {}'.format(df['2'].isnull().sum(), len(df)))
    return df

def plot_minor():
    "Plot minor axis: http://stackoverflow.com/a/21924612/386279"
    ax = plt.gca()
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)


# # Load

# In[ ]:

dow = [u'Mon', u'Tue', u'Wed', u'Thu', u'Fri']
hrs = map(str, range(24))


# In[ ]:

# loc_id = '089-3572'
# df0 = pd.read_msgpack('data/089-3572/all_0.msg')
# dfnor = read('135-6287', sheet=1)
dfnor = read('121-5114', sheet=1).assign(Date=lambda df: df.Date.map(mc('date')))


# In[ ]:

smgdays = range(27, 32)
smg = dfnor.query('Weekday & Year == 2014 & Day_of_year == @smgdays').set_index('Day_of_week')[hrs]
smg.T.plot(figsize=(20, 10), linewidth=4, #fontsize=20,
           title=('The week of Carmageddon, when Atlanta collectively lost its mind'))


# ### Peak hour
# I was curious to see whether the peak traffic hour evolves over the year, but sampling the first 5 weekdays in March, June and October indicate that it peaks somewhat consistently at pm throughout the year.

#     def plot_peak(df, **kw):
#         df[just_nums(df)].T.plot(figsize=(10, 5), **kw)

# In[ ]:

dfpeak = dfnor.query('Weekday & Year == 2014 & Month == [3, 6, 10]').set_index('Date').copy()
dfpeak['Mday'] = dfpeak.Month.map(str) + dfpeak.Day_of_week
dfpeak = dfpeak.drop_duplicates('Mday')


# dim = 1, 3
# 
# [(i, j) for i in range(1, 1 + dim[0]) for j in range(1, 1 + dim[1])]

# In[ ]:

def plotby(df, by=None, f=None, nrows=1, ncols=1):
    ""
    gb = df.groupby(by)
    n = len(gb)
    return [f(i, gbk, dfgb, n, nrows=nrows, ncols=ncols) for i, (gbk, dfgb) in enumerate(gb, 1)]

def plot_days(i, month, dfgb, n, nrows=1, ncols=1, hrs=hrs):
    plt.subplot(nrows, ncols, i)
    plt.title(calendar.month_name[month])
    dfplt_ = dfgb[hrs].T
    for c in dfplt_:
        plt.plot(dfplt_.index, dfplt_[c])
    plt.legend(list(dfplt_), loc='best')

plt.figure(figsize=(20, 5))
plotby(dfpeak, by='Month', f=partial(plot_days, hrs=hrs[10:20]), nrows=1, ncols=3);


# I did note, though, 

# In[ ]:

dfpeak[hrs].T.plot(figsize=(20, 5))
plot_minor()


# ## Cluster 

# In[ ]:

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans, spectral_clustering, AgglomerativeClustering
from sklearn.manifold import TSNE


#     import toolz.curried as z
#     import itertools as it

# In[ ]:

get_ipython().magic(u'pinfo SpectralClustering')


# In[ ]:

Xdf_ = dfnor.dropna(axis=0, how='any', subset=hrs).copy()  #.query('Weekday')
Xdf_['Total'] = Xdf_[hrs].sum(axis=1)
Xdf = Xdf_[hrs]  #.query('Weekday')
X = Xdf


# In[ ]:

Xs = StandardScaler().fit_transform(X)


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"np.random.seed(3)\nts = TSNE(n_components=2)\nXt = ts.fit_transform(X)\nXdf_['T1'], Xdf_['T2'] = zip(*Xt)")


# clust_color = it.cycle(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
# clust_color = dict(zip(set(xdt), clust_color))
# ccol = map(clust_color.get, xdt)
# 
# cp = dict(zip(clusts, sns.color_palette('Set2', n_colors=len(clusts))))
# 

# In[ ]:




# In[ ]:

import sklearn
sklearn.__version__


# In[ ]:

dbts = DBSCAN(eps=1.5)
Xdf_['Tclust'] = dbts.fit_predict(Xt)
# Xdf_['Dclusts'] = DBSCAN(eps=302).fit_predict(X)
# Xdf_['Dclusts'] = DBSCAN(eps=302).fit_predict(X)



# In[ ]:

np.random.seed(3)
# Xdf_['Dclusts'] = SpectralClustering(n_clusters=4, random_state=2, eigen_solver='arpack', assign_labels='discretize').fit_predict(Xt)
Xdf_['Dclusts'] = AgglomerativeClustering(n_clusters=4, linkage='average').fit_predict(Xt)
# Xdf_['Dclusts'] = KMeans(n_clusters=4, random_state=5).fit_predict(Xt)

# clusts = sorted(set(xdt))
# cp = dict(zip(clusts, sns.color_palette('Set2', n_colors=len(clusts))))
# Xdf_['Col'] = ccol = map(cp.get, xdt)

Xdf_.Dclusts.value_counts(normalize=0)


# In[ ]:

Xdf_['Clust_size'] = Xdf_.Tclust.astype(str) + ': n=' + Xdf_.groupby('Tclust').Date.transform(len).map(str)
Xdf_['Dclust_size'] = Xdf_.Dclusts.astype(str) + ': n=' + Xdf_.groupby('Dclusts').Date.transform(len).map(str)


# In[ ]:

sns.set(font_scale=1.5)
# with sns.color_palette('Set2', Xdf_.Tclust.nunique()):
with sns.color_palette('colorblind', Xdf_.Dclust_size.nunique()):
    sns.lmplot("T1", "T2", data=Xdf_, hue='Dclust_size', fit_reg=False, size=8, aspect=2)
plt.xlim(-20, 22)
yo, yi = plt.ylim()


# In[ ]:

del Xdf_['Tclust']


# In[ ]:

sns.lmplot("T1", "T2", data=Xdf_.query('Dclusts == -1'), fit_reg=False, size=4, aspect=2)
plt.xlim(-20, 22)
plt.ylim(yo, yi);


# In[ ]:

sns.palplot(sns.color_palette('Set2', 10))


# It looks like DBSCAN found 4 main clusters (0-3), and identified several smaller ones on the outskirts of these (4-8). 
# 

# Looking at the day of week distribution of the clusters should give a good first pass at dissecting what distinguishes them:

# In[ ]:

Xdf_.groupby(['Tclust', 'Day_of_week']).size().unstack().fillna(' ')


# Clusters 1 and 0 look like they account pretty well for Saturday and Sunday, respectively. 

# In[ ]:

for c, clustdf in Xdf_.set_index('Date').groupby('Tclust'):
    1


# In[ ]:

Xdf_.groupby(['Tclust', 'Day_of_week']).size().unstack().fillna(0).apply(np.log)


# In[ ]:

sns.clustermap(Xdf_.groupby(['Clust_size', 'Day_of_week']).size().unstack().fillna(0), annot=True, linewidths=.5)


# In[ ]:

clustdf


# In[ ]:

get_ipython().magic(u'pinfo2 plot_days')


# 

# In[ ]:

def plot_days2(i, k, dfgb, n, nrows=1, ncols=1, hrs=hrs):
    print('nrows: {},ncols: {},i: {}'.format(nrows, ncols, i))
    plt.subplot(nrows, ncols, i)
    plt.title(k)
    dfplt_ = dfgb[hrs][:10].T
    for c in dfplt_:
        plt.plot(dfplt_.index, dfplt_[c])
    plt.legend(list(dfplt_), loc='best')


# In[ ]:




# In[ ]:




# In[ ]:

Xdf_.set_index('Date')


# In[ ]:

plt.figure(figsize=(20, 20))

plotby(Xdf_.set_index('Date'), by='Clust_size', f=plot_days2, nrows=4, ncols=3);


# In[ ]:

def plot_months(i, month, dfgb, n, nrows=1, ncols=1):
    plt.subplot(nrows, ncols, i)
    plt.title(calendar.month_name[month])
    dfplt_ = dfgb[hrs].T[10:20]
    for c in dfplt_:
        plt.plot(dfplt_.index, dfplt_[c])
    plt.legend(list(dfplt_), loc='best')

plt.figure(figsize=(20, 5))
plotby(dfpeak, by='Month', f=plot_months, nrows=1, ncols=3);


# In[ ]:

# plt.subplot(nrows, ncols, i)
# plt.title(calendar.month_name[month])
dfgb = clustdf

dfplt_ = dfgb[hrs].T #[10:20]
for c in dfplt_:
    plt.plot(dfplt_.index, dfplt_[c])
plt.legend(list(dfplt_), loc='best')


# In[ ]:

clustdf


# In[ ]:

dd = clustdf[hrs].T
dd.plot()


# In[ ]:

Xmeta = Xdf_.drop(just_nums(Xdf_), axis=1)


# In[ ]:

Xmeta.query('Tclust == -1')[:10]


# In[ ]:

Xmeta.query('Tclust == 5')[:10]


# In[ ]:

.query('Tclust == -1')[:10]


# In[ ]:

sns.


# In[ ]:

Xdf_[:2]


# In[ ]:

Xt


# In[ ]:

ccol


# In[ ]:

plt.figure(figsize=(16, 10))
plt.scatter(*zip(*Xt), alpha=.75, color=ccol)
plt.legend(clusts)


# In[ ]:

Xdf_.query('T1 < -10 & T2 < -5').Day_of_week.value_counts(normalize=0)


# In[ ]:

Xdf_.query('T1 < -10 & T2 < -5') #.Day_of_week.value_counts(normalize=0)


# In[ ]:

DataFrame(Xs)[:30].T.plot()


# In[ ]:




# In[ ]:

ss[ss >= 0].value_counts(normalize=0).tolist()


# In[ ]:

def try_dbparam_(eps, X):
    db = DBSCAN(eps=eps)
    ks = db.fit_predict(X)
    ss = Series(ks)
    return ss[ss >= 0].nunique(), (ss < 0).sum(), (ss[ss >= 0].value_counts().tolist() + [0])[0]

def try_dbparam(X, grids):
    params = try_dbparam.params = {e: try_dbparam_(e, X) for e in grids}
    sp = Series(params)
    nz, z, biggest = zip(*sp)
    mkseries = partial(Series, index=sp.index)
    return Series(nz, index=sp.index), Series(z, index=sp.index), mkseries(biggest)


# In[ ]:

nz, z = zip(*sp)
nz, z


# In[ ]:

sp = Series(try_dbparam.params)
sp


# In[ ]:

try_dbparam.params


# In[ ]:

db = DBSCAN(eps=150)
ks = db.fit_predict(X)
ss = Series(ks)


# In[ ]:

ss.value_counts(normalize=0)


# In[ ]:

get_ipython().magic(u'time sparam, zparam, big = try_dbparam(135, 500, X)')


# In[ ]:

1.00000000e-001


# In[ ]:

np.logspace(-1, 2, 50)


# In[ ]:

zparams


# In[ ]:

# %time sparams, zparams, bigs = try_dbparam(Xs, np.logspace(0, 1, 50))
get_ipython().magic(u'time sparams, zparams, bigs = try_dbparam(Xs, np.linspace(1, 2, 100))')


# In[ ]:

np.log10(3)


# In[ ]:

n1 = 1
n = 620
plt.figure(figsize=(16, 10))
sparams[:n].plot()
zparams[:n].plot()
bigs[:n].plot()
plot_minor()
plt.legend(['Nclusters', 'None', 'Big'])


# In[ ]:

params


# In[ ]:

Series(params)


# In[ ]:




# In[ ]:

plt.figure(figsize=(16, 10))
sparams = Series(params)
sparams[:350].plot()
plot_minor()


# In[ ]:

sparams


# In[ ]:

db = DBSCAN(eps=300)
ks = db.fit_predict(X)

Series(ks).value_counts(normalize=0)


# In[ ]:

Xdf_[:2]


# In[ ]:

Xs


# In[ ]:

Data


# In[ ]:

dfnor[:2]


# In[ ]:

get_ipython().magic(u'pinfo StandardScaler')


# In[ ]:

ls


# In[ ]:

dfpeak.groupby('Month').plot()


# In[ ]:

g = sns.FacetGrid(dfpeak[just_nums(dfpeak)] + ['Month'], col='Month')
g.map(plt.plot, figsize=(10, 5))


# In[ ]:

plot_peak(dfpeak)


# In[ ]:

dfpeak


# In[ ]:

df2 = dfnor[:].query('Weekday & Year == 2014 & Month > 2').set_index('Date').copy()
hrs2 = df2[just_nums(df2)]
hrs2[:5].T.plot(figsize=(20, 10))
plot_minor()


# In[ ]:

# plt.figure(figsize=(16, 10))

# hrs.iloc[range(21, 34)].T.plot(figsize=(20, 10))
hrs2[:5].T.plot(figsize=(20, 10))

ax = plt.gca()
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

ax.grid(b=True, which='major', color='w', linewidth=1.0)
ax.grid(b=True, which='minor', color='w', linewidth=0.5)


# In[ ]:




# In[ ]:




# In[ ]:

df6 = dfnor[:].query('Weekday & Year == 2014 & Month == 6').set_index('Date').copy()
hrs6 = df6[just_nums(df6)]
hrs6[:5].T.plot(figsize=(10, 5))

plot_minor()


# In[ ]:

sns.axes_style()


# In[ ]:

weirds = [0, 20]


# In[ ]:




# In[ ]:




# In[ ]:

hrs.iloc[weirds + range(16,20)].T.plot(figsize=(20, 10))


# In[ ]:

df = drop_exc(df0[:], 7).query('Weekday').set_index('Date').copy()
df['Week_avg'] = df.groupby('Week_of_year')['7'].transform('mean')
df['Week_diff'] = df['7'] - df.Week_avg
df[:2]


# In[ ]:

plt.figure(figsize=(16, 10))
df.Week_diff.plot()
df['7'].plot()


# In[ ]:

# pc3 = np.percentile(df['7'], 3)
df[df['7'] < np.percentile(df['7'], 3)].reset_index().set_index(['Year', 'Day_of_year'])


# In[ ]:

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# In[ ]:

bday_us.holidays


# In[ ]:

bday_us.holidays


# In[ ]:

df['7'].plot()


# In[ ]:

df.groupby('Day_of_week')[['Week_diff', '7']].agg(['mean', 'median']).ix[dow]


# In[ ]:

df.Day_of_week[:5].tolist()


# In[ ]:

sns


# In[ ]:

plt.figure(figsize=(16, 10))
sns.violinplot(x='Day_of_week', y='Week_diff', data=df, )


# In[ ]:

df.groupby('Week_of_year')['7'].transform('mean')


# In[ ]:

# df0.query('Weekday').groupby(['Year', 'Day_of_year'])['7'].mean().plot()  # Day_of_week
plt.figure(figsize=(16, 10))
drop_exc(df0[:], 7).query('Weekday').groupby(['Date'])['7'].mean().plot()
rotate(70)


# In[ ]:

get_ipython().system(u'cat data/135-6287/description.txt')


# In[ ]:

drop_exc(df0.query('Weekday'), 7)[-100:]


# - peak hour

# In[ ]:




























# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

plt.plota


# In[ ]:

sheetnames = (u'Combined', u'Dir E', u'Dir W')


# In[ ]:

def check_sheetnames(dir):
    dirglob = glob.glob(join(dir, '*.xlsx'))
    snames = {tuple(xlrd.open_workbook(fn).sheet_names()) for fn in dirglob}
    assert snames == {sheetnames}, "Different sheet names: {}".format(snames)


# In[ ]:

# check_sheetnames('data/121-5450')


# In[ ]:

def load_month(yr, mth):
    

