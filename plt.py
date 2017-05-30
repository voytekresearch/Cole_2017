"""
plt.py
Some plotting functions
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def scatt_2cond(x, y, ms=12,
                lims=None, ticks=None,
                xlabel='', ylabel='',
                figsize=(5, 5),
                returnax=False):
    if lims is None:
        lims = (np.min(np.hstack((x, y))), np.max(np.hstack((x, y))))
    if ticks is None:
        ticks = lims

    plt.figure(figsize=figsize)
    plt.plot(x, y, 'k.', ms=ms)
    plt.plot(lims, lims, 'k-')

    plt.xlim(lims)
    plt.xticks(ticks, size=15)
    plt.xlabel(xlabel, size=20)

    plt.ylim(lims)
    plt.yticks(ticks, fontsize=15)
    plt.ylabel(ylabel, size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()


def scatt_corr(x, y, ms=12,
               xlim=None, ylim=None,
               xticks=None, yticks=None,
               xlabel='', ylabel='',
               showrp=False, ploc=(0, 0), rloc=(0, 1), corrtype='Pearson',
               showline=False,
               figsize=(5, 5),
               returnax=False):
    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y), np.max(y))
    if xticks is None:
        xticks = xlim
    if yticks is None:
        yticks = ylim

    plt.figure(figsize=figsize)
    plt.plot(x, y, 'k.', ms=ms)

    if showline:
        from tools.misc import linfit
        linplt = linfit(x, y)
        plt.plot(linplt[0], linplt[1], 'k--')

    if showrp:
        if corrtype == 'Pearson':
            r, p = sp.stats.pearsonr(x, y)
        elif corrtype == 'Spearman':
            r, p = sp.stats.spearmanr(x, y)
        ax = plt.gca()
        ax.text(rloc[0], rloc[1], '$r^2 = $' +
                np.str(np.round(r**2, 2)), fontsize=15)
        ax.text(ploc[0], ploc[1], '$p = $' +
                np.str(np.round(p, 3)), fontsize=15)

    plt.xlim(xlim)
    plt.xticks(xticks, size=20)
    plt.xlabel(xlabel, size=20)

    plt.ylim(ylim)
    plt.yticks(yticks, fontsize=20)
    plt.ylabel(ylabel, size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()
