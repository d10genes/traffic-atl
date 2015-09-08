
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic(u'javascript', u'', u"IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[2]:

from __future__ import print_function, division
from os.path import join, exists, dirname
import os
import re
import errno
from functools import partial
import datetime as dt
from dateutil import rrule
from datetime import datetime, timedelta
from collections import Counter

import requests
from bs4 import BeautifulSoup


#     s = requests.Session()
#     s.get('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#')
#     s.id = 43946

# In[8]:

def gen_session(siteid_sess):
    s, url = gen_session_(siteid_sess)
    html = requests.get(url).content
    s.id = get_siteid(html)
    s.loc_id = siteid_sess
    dirname = gen_name('', '', siteid_sess, dironly=1)
    mkdirs(dirname)
    descfile = join(dirname, 'description.txt')
    if not exists(descfile):
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

def get_siteid(html):
    siteid_pat = re.compile(r"doShowRawMonth\('(\d+)', \d+, \d+\);")
    ctr = Counter(siteid_pat.findall(html))
    return sorted(ctr.items(), key=lambda x: -x[1])[0][0]


# In[6]:

# ss = gen_session('135-6287', 43946)
# ss = gen_session('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#', 43946)
# rr = gen_session('121-0124') 
s400 = gen_session('121-5450') 


# In[7]:

fns = fetchall(dt.date(2012, 1, 1), dt.date(2015, 8, 1), s400)

