from glob import glob
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlrd


def read(site_loc, sheet=0, verbose=True):
    "Read saved excel sheet into dataframe"
    fn = join('data', site_loc, 'all_{}.msg'.format(sheet))
    exfile = glob(join('data', site_loc, '*.xlsx'))[0]

    xl_workbook = xlrd.open_workbook(exfile)
    sheet_names = xl_workbook.sheet_names()
    del xl_workbook

    df = pd.read_msgpack(fn)

    if verbose:
        print('{} => {}'.format(sheet_names, sheet_names[sheet]))
        with open(join('data', site_loc, 'description.txt')) as f:
            print(f.read())
        print('Nulls: {} / {}'.format(df['2'].isnull().sum(), len(df)))
    return df


def rotate(deg=90):
    "Rotate xlabels by `deg` degrees"
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=deg)


def plot_minor():
    "Plot minor axis: http://stackoverflow.com/a/21924612/386279"
    ax = plt.gca()
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
