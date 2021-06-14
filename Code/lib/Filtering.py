from lib.Fitting import *
import numpy as np
from scipy.ndimage import gaussian_filter1d as gaussFilter

# Author: Damian Mendroch,
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Filtering methods
"""


def filterProfile(x, y, F):
    """
    semi-automatic selective filtering of 1D data. Data is filtered using polynomial splines at regions
    with d^2 y /d x^2 < tr, with tr being a threshold value, otherwise data is copied to output.
    Addtional regions can be created using the xb array in the F dictionary.

    :param x: abscissa values starting at 0 (numpy 1D array)
    :param y: ordinate values (numpy 1D array)
    :param F: filter properties dictionary (see code for more details)
    :return: filtered curve, second derivative and filtered second derivative (all numpy 1D arrays)
    """

    # F dictionary includes:
    #  fc1: gaussian filter constant for pre-filtering of y before differentiation
    #  fc2: gaussian filter constant for post-filtering after differentiation
    #  n: polynomial spline order,
    #  tr: threshold for section division,
    #  xb: x-values for additional sections division coordinates

    # work on copy
    y_out = y.copy()

    # pre- and post-filtering of absolute second derivative
    dx = x[1] - x[0] # step size
    y_f = gaussFilter(y, F['fc1']) #filtered curve
    y2 = np.abs(np.concatenate(([0], np.diff(np.diff(y_f)), [0]))) / dx**2 # abs(d^2 y/dx^2)
    y2_f = gaussFilter(y2, F['fc2']) # filtered abs(d^2 y/dx^2)

    # find section beginnings/endings
    xal = np.append(y2_f, y2_f[-1]) # left shifted y2_f
    xar = np.insert(y2_f, 0, y2_f[0]) # right shifted y2_f
    xa = np.logical_xor(xal > F['tr'], xar > F['tr']).nonzero()[0] # find threshold intersections using xor

    # convert additional sections points from x-values to indices
    xb = np.round(F['xb']/dx).astype(int)

    # add section beginning
    if y2_f[0] < F['tr']:
        xa = np.concatenate(([0], xa))
    # add section ending
    if y2_f[-1] < F['tr']:
        xa = np.concatenate((xa, [y2_f.shape[0]]))

    # no sections -> nothing to filter
    if xa.shape == 0:
        return y_out

    # check for additional section points that lie within ignored region
    if xb.shape[0] > 0:
        xbp = np.where(y2_f[xb] > F['tr'])[0]
        if xbp.shape[0] > 0:
            print("Ignoring section points r =", xb[xbp] * dx, "that lie within ignored region")
            xb = np.delete(xb, xbp)

        # duplicate xb points (we need one value for beginning and ending) and resort the sections
        xa = np.sort(np.concatenate((xa, xb, xb)))

    # section filtering
    for n in np.arange(0, xa.shape[0], 2):

        xan = np.arange(xa[n], xa[n+1]) # n-th section range
        order = min(xan.shape[0]//4, F['n']) # reduce degree at small sample size

        if xa[n] == 0:  # ensure dy/dx = 0 at x = 0 by symmetrical polynomial for first section
            s = SymPolyRegression(x[xan], y[xan], order)
        else: # else normal polynomial
            s = PolyRegression(x[xan], y[xan], order)

        # write to output
        y_out[xan] = Poly(x[xan], s)

    return y_out, y2, y2_f
