
# coding: utf-8

# As someone constantly looking for ways to waste less time in traffic, I'm always interested in what I can learn about Atlanta traffic patterns. After a bunch of searching, I finally found a way to access numerical traffic data at http://geocounts.com/gdot/, in the form of hourly vehicle counts for several years at different locations Georgia. While I would prefer a way to measure travel time trends, I figured it would be worthwhile to see what could be gleaned from volume measurements. 
# 
# In this post I take a shot at visualizing the volume trends (downloaded from the **load.ipynb** notebook) from a time-series and then clustering point of view using tools from the python data analysis stack. First for the imports and data loading...

# In[1]:

get_ipython().run_cell_magic(u'javascript', u'', u"IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[2]:

from __future__ import print_function, division
from os.path import join, exists, dirname
from functools import partial
from glob import glob
import datetime as dt
import calendar
from itertools import count
from operator import methodcaller as mc

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import toolz.curried as z

from traffic_utils import read, rotate, plot_minor

pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
get_ipython().magic(u'matplotlib inline')
sns.set_palette(sns.color_palette('colorblind', 5))


# In[3]:

dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
adow = dow + ['Sat', 'Sun']
hrs = map(str, range(24))


# Of the several sensor sites that were available, I picked one that looked most complete (many had hours and even months missing). This data represents the north-bound traffic flow on Roswell Road.

# In[4]:

dfnor = read('121-5114', sheet=1, verbose=0).assign(Date=lambda df: df.Date.map(mc('date')))
dfnor[:2]


# Each row of the data represents a day, and there are 24 numerical columns (0-23) representing the number of cars passing the point at the given hour. As a sample of what a week of data looks like, here is the [infamous last week of January 2014](http://news.yahoo.com/atlanta-s--snowpocalypse--turned-ordinary-commutes-into-chaos-and-confusion-174131033.html). Each line represents a separate day, and the $x$ and $y$ axes represent the hour and traffic counts, respectively. There is a general trend of high peaks around noon and 5pm (presumably for lunch and return from work time) and a smaller morning peak at around 9 (this is northbound and north of Atlanta while most traffic will be going down towards the city in the morning).

# In[5]:

smgdays = range(27, 32)
smg = dfnor.query('Weekday & Year == 2014 & Day_of_year == @smgdays').set_index('Day_of_week')[hrs]
smg.T.plot(figsize=(20, 10), linewidth=4, #fontsize=20,
           title=('The week of Carmageddon, when Atlanta collectively lost its mind'))


# ### Peak hour
# I was curious to see whether the peak traffic hour evolves over the year, but sampling the first 5 weekdays in March, June and October indicate that it peaks somewhat consistently on the evening throughout the year at 8am, noon and 5pm.

# In[6]:

dfwkday = dfnor.query('Weekday').set_index('Date').copy()
dfpeak = dfwkday.query('Year == 2014 & Month == [3, 6, 10]').copy()
dfpeak['Mday'] = dfpeak.Month.map(str) + dfpeak.Day_of_week
dfpeak = dfpeak.drop_duplicates('Mday')


# In[7]:

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
plotby(dfpeak, by='Month', f=partial(plot_days, hrs=hrs[5:20]), nrows=1, ncols=3);


# And, perhaps not surprisingly, there appears to be a lot less variation from 1-4pm

# In[8]:

dfpeak[hrs].T.plot(figsize=(20, 5))
plot_minor()


# Looking at how the peak hours change throughout the year produced a lot of spiky behavior, which taking the moving average helped with:

# In[9]:

sns.set(font_scale=1.5)
plt.figure(figsize=(20, 6))
plot_peak_smoothed = lambda x: pd.rolling_mean(x, 5).plot()
get_peak = lambda x: x.T.idxmax().dropna().map(int)

am = get_peak(dfwkday[['6', '7', '8', '9', '10']])
lunch = get_peak(dfwkday[['10', '11', '12', '13', '14']])
pm = get_peak(dfwkday[['14', '15', '16', '17', '18', '19']])

plot_peak_smoothed(am)
plot_peak_smoothed(lunch)
plot_peak_smoothed(pm)

year_line = lambda yr: plt.plot([dt.date(yr, 1, 1), dt.date(yr, 1, 1)], [8, 18], '-.', linewidth=.7, c=(0,0,0))
map(year_line, [2013, 2014, 2015])

rotate(75)
plt.xlim()
plot_minor()


# The intuition that peak hour doesn't vary much looks largely correct. They look pretty constant throughout the year, though there appears to be a discernible dip in the peak afternoon time (and corresponding rise in morning and lunch times) around holidays such as Christmas, New Years, Independence day and Labor Day. It looks like the summer months also have a slightly later peak in the morning, perhaps because the school schedule allows for a later departure. There's a lot more variation, though, so it's hard say with a lot of precision.
# 
# These patterns are perhaps more clear with all the years overlaid on each other:

# In[10]:

@z.curry
def new_year(newyear, d):
    "Given date object, return copy with same date, but year `newyear`"
    (_, m, day) = d.timetuple()[:3]
    return dt.date(newyear, m, day)


plt.figure(figsize=(16, 6))
for yr, yeardf_ in dfwkday.groupby('Year'):
    yeardf = yeardf_.copy()
    yeardf.index = yeardf.index.map(new_year(2000))  # Changing all years to 2000 for easier overlay
    
    am = get_peak(yeardf[['6', '7', '8', '9', '10']])
    lunch = get_peak(yeardf[['10', '11', '12', '13', '14']])
    pm = get_peak(yeardf[['14', '15', '16', '17', '18', '19']])
    
    plot_peak_smoothed(am)
    plot_peak_smoothed(lunch)
    plot_peak_smoothed(pm)
rotate(75)


# ## Dimensionality reduction and clustering

# I was wanting to see how much the traffic volume for each day cluster, using the hourly volume as features. I had a hard time finding obvious clusters on the raw data, and took a detour and tried to visualize the daily volume. PCA didn't reveal anything that stood out to me, but running [TSNE](http://lvdmaaten.github.io/tsne/) revealed some interesting patterns

# In[11]:

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans, spectral_clustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[12]:

Xdf_ = dfnor.dropna(axis=0, how='any', subset=hrs).copy()
Xdf_['Total'] = Xdf_[hrs].sum(axis=1)
X = Xdf_[hrs]


# In[13]:

get_ipython().run_cell_magic(u'time', u'', u"np.random.seed(3)\nts = TSNE(n_components=2)\nXt = ts.fit_transform(X)\nXdf_['T1'], Xdf_['T2'] = zip(*Xt)")


# While much slower to run than PCA, TSNE is a popular way to reduce high-dimensional data into 2 or 3 dimensions for informative visualization. With our dataset, it does a good job of cramming the 24 features (traffic volume for each hour of the day) down into just 2 while preserving important characteristics of the data.
# 
# Here we see how well it spreads the data out, and seems to partition it into 4 natural clusters, compared to the lumpier PCA results which basically just leave us with 2 main clusters:

# In[14]:

xpca = DataFrame(PCA(n_components=2).fit_transform(X), columns=['P1', 'P2'])

plt.figure(figsize=(16, 5))
plt.subplot(1,2,1)
plt.title('TSNE')
plt.scatter(Xdf_.T1, Xdf_.T2, alpha=.5, c="#3498db")

plt.subplot(1,2,2)
plt.title('PCA')
plt.scatter(xpca.P1, xpca.P2, alpha=.5, c="#9b59b6");


# After not having much success clustering the raw data, the 2 TSNE dimensions looked like better candidates for clustering, and hierarchical clustering seemed to give pretty good results without much tuning:

# In[15]:

Xdf_['Tclust'] = AgglomerativeClustering(n_clusters=4, linkage='average').fit_predict(Xt)

# Label each cluster by how many samples are in it
Xdf_['Clust_size'] = Xdf_.Tclust.astype(str) + ': n=' + Xdf_.groupby('Tclust').Date.transform(len).map(str)

with sns.color_palette('colorblind', Xdf_.Clust_size.nunique()):
    sns.lmplot("T1", "T2", data=Xdf_, hue='Clust_size', fit_reg=False, size=8, aspect=2)
plt.xlim(-20, 22)
plt.title('Clustering in TSNE space');


# It looks like AgglomerativeClustering was able to find the 4 main clusters pretty well; looking at the day of week distribution of the clusters gave a good first pass at dissecting what distinguishes them:

# In[16]:

Xdf_.groupby(['Tclust', 'Day_of_week']).size().unstack().fillna(' ')[adow]


# It looks like the clusters quite neatly decompose into day-of-the-week categories: Saturday (#3), Sunday (#1), Friday (#2) and the rest of the weekdays (#0). I was a little surprised that Fridays are so cleanly segregated from the rest of the weekdays, and initially reasoned from annecdotes that it must be less busy, as it seems colleagues tend to take them off more and traffic seems much lighter compared to the other 4 work days.
# 
# Just to verify I aggregated average counts by day and found that Fridays don't actually have a discernably higher average traffic flow from 2-7pm. Looking at the medians also didn't give any evidence it's just some outliers pulling up the average.

# In[17]:

gohomehrs = hrs[14:20]
rushhour_pm = Xdf_.query('Weekday')[gohomehrs + ['Day_of_week']].assign(Total=lambda x: x[gohomehrs].sum(axis=1))

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))
sns.barplot(x='Day_of_week', y='Total', data=rushhour_pm, ax=ax1);
sns.barplot(x='Day_of_week', y='Total', data=rushhour_pm, estimator=np.median, ax=ax2);


# But plotting a sample of Fridays and non-Fridays shows a bit of a nuanced diffence:

# In[18]:

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))
sampkwds = dict(n=30, random_state=1)
pltkwds = dict(legend=False, alpha=.25)
nofrisamp = dfnor.query('Weekday & Day_of_week != "Fri"').sample(**sampkwds)[hrs].T
frisamp = dfnor.query('Day_of_week == "Fri"').sample(**sampkwds)[hrs].T

nofrisamp.plot(ax=ax1, title='Mon-Thurs',**pltkwds)
frisamp.plot(ax=ax2, title='Friday', **pltkwds);


# A couple of major shape differences initially jumped out at me. While traffic drops to about 200 by midnight on work nights, it looks like it's about twice that on Friday nights, and the lull between lunch and 5pm is less pronounced on Fridays. Plotting aggregated median volume at 2pm and 11pm show that this seems to hold for the days not in these samples.

# In[19]:

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))
lull_night = Xdf_.query('Weekday')[['14', '23', 'Day_of_week']] #.assign(Total=lambda x: x['14'].sum(axis=1))

sns.barplot(x='Day_of_week', y='14', data=lull_night, estimator=np.median, ax=ax1);
sns.barplot(x='Day_of_week', y='23', data=lull_night, estimator=np.median, ax=ax2);


# And just for kicks, I thought I'd reverse the hours and look at the normalized cumulative sum to see if anything stood out. Beyond the higher midnight volume, though, nothing really stood out.

# In[20]:

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))

nofrisamp[::-1].cumsum().apply(lambda x: x / x.max() * 100).plot(ax=ax1, title='Mon-Thurs',**pltkwds)
frisamp[::-1].cumsum().apply(lambda x: x / x.max() * 100).plot(ax=ax2, title='Friday', **pltkwds);


# Anyways, back to those clusters, it looked useful to relabel them according to the most common day in the cluster, and examine the outliers

# In[21]:

Xdf_['Day_label'] = Xdf_.Tclust.map({0: 'Week', 1: 'Sun', 2: 'Fri', 3: 'Sat'})
with sns.color_palette('colorblind', Xdf_.Day_label.nunique()):
    sns.lmplot("T1", "T2", data=Xdf_, hue='Day_label', fit_reg=False, size=8, aspect=2)
plt.xlim(-20, 22);


# Of the possible pairings for the four main clusters, the Saturday and Sunday clusters look like they may be hardest to disambiguate. While they can be separated pretty easily on the first dimension, there's a huge amount of overlap in the second. While I don't know of any way to directly interpret TSNE results, it looks like T2 represents something predictive of weekend-ness.
# 
# 
# Zooming in on the large weekday cluster, labelling with the *actual* day of the week shows an interesting density shift corresponding to the order of the day: the lower left is dominated by Mondays, which transitions to Tuesdays, followed by Wednesdays and ending with Fridays (this trend can also be seen in the successively higher traffic volume at 2pm and 11pm in the median bar plots above).

# In[22]:

C0 = (Xdf_.query('Tclust == 0 & Day_of_week != "Fri"')
      .assign(Daynum=lambda x: x.Day_of_week.map(dict(zip(dow, count()))))
      .sort('Daynum'))
with sns.color_palette('Paired', C0.Day_of_week.nunique()):
    sns.lmplot("T1", "T2", data=C0, hue='Day_of_week', fit_reg=False,
               size=8, aspect=2, hue_order=dow[:-1], scatter_kws=dict(s=70, alpha=.9))
plt.xlim(-15, 15);


# This ordered shift is more clear in a real density plot, particularly in the *T2* dimension

# In[23]:

with sns.color_palette('Paired', C0.Day_of_week.nunique()):
    g = sns.JointGrid("T1", "T2", C0, size=10)
    for day, daydf in C0.groupby('Day_of_week', sort=False):
        sns.kdeplot(daydf["T1"], ax=g.ax_marg_x, legend=False)
        sns.kdeplot(daydf["T2"], ax=g.ax_marg_y, vertical=True, legend=False)
        g.ax_joint.plot(daydf.T1, daydf.T2, "o", ms=5)
    plt.legend(dow)


# ## Wrapup
# 
# This post has been a brief exploration into Atlanta traffic patterns. My overall goal was to visually discover characteristics of traffic volume and how these vary over time. I found tracking the peak hour over time to be useful, as well as clustering the data in TSNE space and inspecting what is behind the differences in those clusters. I hope I have also showed useful aspects and capabilities that Python data libraries like pandas, sklearn and seaborn have to offer.
# 
# There are many more  ways to do this kind of analysis which may be worth looking into. Feel free to download the notebook and do some of your own exploring.
# 
