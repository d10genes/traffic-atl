
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic(u'javascript', u'', u"IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from __future__ import print_function, division
from os.path import join, exists, dirname, basename
import os
import re
import errno
from functools import partial
import datetime as dt
from datetime import datetime, timedelta
import calendar
from collections import Counter
from glob import glob
import itertools as it
from operator import attrgetter as prop

import requests
from bs4 import BeautifulSoup
from dateutil import rrule, relativedelta
from dateutil.relativedelta import relativedelta 

import pandas as pd
import toolz.curried as z

pd.options.display.notebook_repr_html = False
pd.options.display.width = 130


#     s = requests.Session()
#     s.get('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#')
#     s.id = 43946

# In[ ]:

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


# In[ ]:

ss = gen_session('135-6287')
# ss = gen_session('http://trafficserver.transmetric.com/gdot-prod/tcdb.jsp?siteid=135-6287#', 43946)
# rr = gen_session('121-0124')  # empty
# s400 = gen_session('121-5450')
# rritp = gen_session('121-5114') # big gaps
# rr = gen_session('121-0124')


# In[ ]:

fns = fetchall(dt.date(2011, 12, 1), dt.date(2015, 9, 1), ss)


# ## Read Excel files 

# In[ ]:

p = print


# In[ ]:

def cols2hrs24(df):
    "Convert columns from `12:00 am,  1:00 am, ...11:00 pm` to `0, 1, ...23`"
    hrs = z.pipe(range(1, 13), it.cycle, z.drop(11), z.take(12), list)
    hrs24 = ['{}:00 {}'.format(hr, half) for half in ('am', 'pm') for hr in hrs]
    assert all(df.columns[2:] == hrs24), "Expecting columns of form `12:00 am,  1:00 am, ...11:00 pm`"
    return df.rename(columns=dict(zip(hrs24, map(str, range(24)))))


def read_excel(fname, sheetname=0):
    # URL scheme subtracted a month from actual contents; correct this
    pat = re.compile(r'(\d{4})_(\d{1,2})\.xlsx')
#     /p fname
#     p(basename(fname))
    [(year_, month_)] = pat.findall(basename(fname))
    prevmonth = dt.date(int(year_), int(month_), 1)
    thismonth = prevmonth + relativedelta(months=1)
    (year, month) = thismonth.timetuple()[:2]
    
    df = pd.read_excel(fname, header=7, sheetname=sheetname)
    df = df.rename(columns={'Unnamed: 0': 'Date', 'Unnamed: 1': 'Day_of_week'})
    
    # Drop aggregate rows/cols
    date_rows = df.Date.str[-1].str.isdigit()
    dregs = df.Date[~date_rows]
    assert (dregs == 'Average Weekday Weekend'.split()).all()
    assert (dregs == df.Date[-3:]).all()
    del df['Total']
        
    # Parse and check days
    df = df[date_rows].copy()
    df['Date'] = pd.to_datetime(str(year) + ' ' + df['Date'].map(str))
    
    if not all(df.Date == pd.Series(get_days(year, month))):
        p(df.Date)
        p(pd.Series(get_days(year, month)))
    assert all(df.Date == pd.Series(get_days(year, month))), "Unexpected date rows"
    
    df = cols2hrs24(df) 
    df['Day_of_year'] = df.Date.map(prop('dayofyear'))
    df['Week_of_year'] = df.Date.map(prop('weekofyear'))
    df['Year'] = df.Date.map(prop('year'))
    df['Weekday'] = ~df.Day_of_week.isin(['Sat', 'Sun'])
    df['Month'] = df.Date.map(prop('month'))
    
    return df # , year, month


get_days = lambda year, month: [dt.datetime(year, month, day) for day in range(1, calendar.monthrange(year, month)[1] + 1)]
# df, year, month = read_excel('data/121-5450/2012_1.xlsx', 2)
df = read_excel('data/121-5450/2012_1.xlsx', 2)
# year, month


# In[ ]:

def collect_dfs(site_loc, sheet=0, verbose=1):
    if hasattr(sheet, '__iter__'):
        return [collect_dfs(site_loc, sheet=s, verbose=verbose) for s in sheet]
    dr = 'data/{}/*.xlsx'.format(site_loc)
    all_dfs = []

    for fname in glob(dr):
        if verbose:
            print(fname)
        all_dfs.append(read_excel(fname, sheetname=sheet))
    df = pd.concat(all_dfs).sort('Date').reset_index(drop=1)
    
    are, shouldbe = zip(*zip(df.Day_of_week, it.cycle(df.Day_of_week[:7])))
    same = pd.Series(are) == pd.Series(shouldbe)
    assert all(same), "Days of week expected to cycle uninterrupted"

    dir = dirname(dr)
    outname = join(dir, 'all_{}.msg'.format(sheet))
    df.to_msgpack(outname)
    return df

# site_dfs = collect_dfs('135-6287', sheet=[0, 1, 2], verbose=0)
# site_dfs = collect_dfs('121-5450', sheet=[0, 1, 2], verbose=0)
site_dfs = collect_dfs('121-5114', sheet=[0, 1, 2], verbose=0)
# site_df0 = collect_dfs('089-3572', sheet=0, verbose=0)
# site_df1 = collect_dfs('089-3572', sheet=1, verbose=0)
# site_df2 = collect_dfs('089-3572', sheet=2, verbose=0)


# In[ ]:

get_ipython().system(u'open data/135-6287/2014_6.xlsx')

