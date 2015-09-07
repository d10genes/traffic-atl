
# coding: utf-8

# In[ ]:

from __future__ import print_function, division

# import arrow
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal
import pypyodbc as odbc
import redis
import toolz.curried as z
xx = list

from ast import literal_eval
from collections import OrderedDict
import csv
import datetime as dt
from decimal import Decimal
from functools import partial as part
import hashlib
import itertools as it
from itertools import starmap, repeat, count
from operator import itemgetter as itg, methodcaller as mc, attrgetter as prop
import os
import re
import sys
import time

import myutils as mu
import pandas_utils as pu
import vutils as vu
pu.psettings(pd, lw=150)


# In[1]:

get_ipython().run_cell_magic(u'javascript', u'', u"IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[2]:

from os.path import join, exists, dirname
import os
import errno
from functools import partial
import datetime as dt
from dateutil import rrule
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


#     s = requests.Session()
#     s.get('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#')
#     s.id = 43946

# In[12]:

def gen_session(siteid_sess, siteid_url):
    s, url = gen_session_(siteid_sess)
    s.id = siteid_url
    s.loc_id = siteid_sess
    dirname = gen_name('', '', siteid_sess, dironly=1)
    mkdirs(dirname)
    descfile = join(dirname, 'description.txt')
    if not exists(descfile):
        html = requests.get(url).content
        txt = get_desc(html)
        print('Writing {}'.format(descfile))
        with open(descfile, 'wb') as f:
            f.write(txt)
    return s 


def gen_session_(siteid_sess):
    s = requests.Session()
    url = 'http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid={}#'.format(siteid_sess)
    s.get(url)
    return s, url


def get_desc(html):
    tdstyle = {u'style': u'text-align: left; width: 100%; padding-left: 6px; padding-top: 0px;'}
    soup = BeautifulSoup(html, 'html.parser')
    td = soup.table.tr.findChild(attrs=tdstyle)
    return td.text

def xls_getter(year, month, siteid=None, session=requests):
    u = ("http://trafficserver.transmetric.com/gdot-prod/exportexec.jsp"
    "?source=tcdb_monthxls&siteid={siteid}&year={year}&month={month}&tfVol=02725170146&tfClass=84923669106")
    url = u.format(year=year, month=month, siteid=siteid)
    r = session.get(url)
    return r    

def to_excel(r, fn, dir=''):
    with open(fn, 'wb') as f:
        for chunk in r.iter_content():
            f.write(chunk)
    return fn

def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

def gen_name(year, month, loc_id=None, dironly=False):
    dir = join('data', str(loc_id))
    if dironly:
        return dir
    return join(dir, '{}_{}.xlsx'.format(year, month))


def fetch(year, month, loc_id=None, session=requests, cache=True):
    loc_id = loc_id or session.loc_id
    fn = gen_name(year, month, loc_id=loc_id)
    #mkdirs(dirname(fn))
    
    if cache and exists(fn):
        print('File pulled. Skipping.')
        return fn
    
    r = xls_getter(year, month, siteid=session.id, session=session)
    to_excel(r, fn)
    return fn


def fetchall(startdate, until, s):
    all_months = ((d.year, d.month) for d in rrule.rrule(rrule.MONTHLY, dtstart=startdate, until=until))
    fns = []
    for yr, mth in all_months:
        try:
            print(yr, mth)
            fn = fetch(yr, mth, session=s)
            fns.append(fn)
        except Exception as e:
            raise(e)
    return fns


# In[10]:

ss = gen_session('135-6287', 43946)
# ss = gen_session('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#', 43946)


# In[13]:

fns = fetchall(dt.date(2012, 1, 1), dt.date(2015, 8, 1), ss)


# In[44]:

d1 = dt.date(2014, 2, 1)
d2 = dt.date(2015, 5, 1)

for d in rrule.rrule(rrule.MONTHLY, dtstart=d1, until=d2):
    print(d.year, d.month)


# In[57]:




# In[ ]:

to_excel(r, '{}_{}_{}.xlsx'.format(43946, 2014, 6))


# In[34]:

get_ipython().system(u"open 'data/'")


# In[ ]:

get_ipython().system(u"open 'data/43946_2014_6.xlsx'")


# In[ ]:

to_excel(r, )


# In[ ]:

r.reason


# In[ ]:

r.request.


# In[ ]:




# In[ ]:



