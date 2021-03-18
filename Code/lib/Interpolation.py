import numpy as np
from scipy import interpolate

"""
Interpolation functions (1D and 2D)

"""


# TODO Bug: Remaining "nan islands". Example: h_data with isnan = [[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]
#  creates remaining nan at isnan[3,3]. Execute function multiple times to solve this issue?
def interpolateNan(h_data):
    """
    2D interpolation of gridded data with missing points (nan values). 2D interpolation of not gridded data
    is extremly slow, so we instead interpolate row and columnwise and use the average of the results at these points.
    This speeds up interpolation by a factor of 10x-100x compared to the function griddata

    :param h_data: z values (2D array)
    :return: h_data with invalid data interpolated (2D array)
    """

    h_data_ix = h_data.copy()
    h_data_iy = h_data.copy()

    x = np.arange(h_data.shape[1])
    y = np.arange(h_data.shape[0])

    # in x direction
    for n in y:
        bad_mask_n = ~np.isfinite(h_data[n,:])

        # only interpolate if more than 10 valid data points and there is missing data to interpolate
        if np.count_nonzero(bad_mask_n) > 0 and np.count_nonzero(~bad_mask_n) > 10:

            # interpolation
            f = interpolate.interp1d(x[~bad_mask_n], h_data_ix[n, ~bad_mask_n])

            # exclude outlying invalid data points, so no extrapolation takes place
            finite = (~bad_mask_n).nonzero()[0]
            bad_mask_n[:finite.min()] = False
            bad_mask_n[finite.max():] = False

            # fill interpolated data
            h_data_ix[n, bad_mask_n] = f(x[bad_mask_n])

    # do the same in y direction
    for n in x:
        bad_mask_n = ~np.isfinite(h_data[:, n])

        if np.count_nonzero(bad_mask_n) > 0 and np.count_nonzero(~bad_mask_n) > 10:
            f = interpolate.interp1d(y[~bad_mask_n], h_data_iy[~bad_mask_n, n])

            finite = (~bad_mask_n).nonzero()[0]
            bad_mask_n[:finite.min()] = False
            bad_mask_n[finite.max():] = False

            h_data_iy[bad_mask_n, n] = f(y[bad_mask_n])

    # initial bad data
    bad_mask = ~np.isfinite(h_data)

    # unfixed data points
    still_bad_x = ~np.isfinite(h_data_ix)
    still_bad_y = ~np.isfinite(h_data_iy)

    # use mean of fixed data in x and y direction where possible
    mean_mask = bad_mask & ~still_bad_x & ~still_bad_y
    h_data_ix[mean_mask] = h_data_ix[mean_mask]/2 + h_data_iy[mean_mask]/2

    # copy data that is fixed in h_data_y but not in h_data_x to h_data_x
    cp_y_mask = bad_mask & still_bad_x & ~still_bad_y
    h_data_ix[cp_y_mask] = h_data_iy[cp_y_mask]

    return h_data_ix


def interpolateProfile(r, profile, r1, method='linear'):
    """
    interpolation of specified ranges, methods of scipy's interp1d are available

    :param r: r vector (1D array)
    :param profile: profile data (1D array)
    :param r1: interpolation range vector, for every range start and end point are given as pairs (1D array)
    :param method: method for scipy's interp1d (string)
    :return: profile with interpolated regions (1D array)
    """

    if not r1: # empty interpolation range
        return profile.copy()

    # work on copies
    r1i = r1.copy()
    profile2 = profile.copy()

    # convert positions to indices
    dr = r[1]-r[0]
    r1i[0::2] = np.floor(r1[0::2]/dr).astype(int) # convert to indices, round lower bounds down
    r1i[1::2] = np.ceil(r1[1::2]/dr).astype(int)  # convert to indices, round upper bounds up

    # check coordinates for validity
    if np.min(r1i) < 1 or np.max(r1i) > r.shape[0] - 2:
        raise Exception("r1 coordinates need to be inside the data range (no extrapolation)")

    if len(r1i) % 2 == 1: # odd number of points
        raise Exception("r1 vector needs to have data pairs")

    # create ranges from indices
    r_r = []
    for n in np.arange(0, len(r1i), 2):
        r_r = np.concatenate((r_r, np.arange(r1i[n], r1i[n+1])))
    r_r = r_r.astype(int)

    # create indices vector excluding interpolate ranges
    r0 = np.arange(r.shape[0])
    r0 = np.delete(r0, r_r)

    # interpolation
    f = interpolate.interp1d(r[r0], profile[r0], method)
    profile2[r_r] = f(r[r_r])

    return profile2


def interp2f(x, y, z, xp, yp, method='linear'):
    """
    faster 2D interpolation on gridded data with linear or nearest neighbor interpolation

    :param x: x coordinate vector (numpy 1D array)
    :param y: y coordinate vector (numpy 1D array)
    :param z: z values (numpy 2D array)
    :param xp: numpy 1D arrays holding the interpolation points x coordinates
    :param yp: numpy 1D arrays holding the interpolation points y coordinates
    :param method: "linear" or "nearest" (string)
    :return: interpolated values as 1D array
    """

    # check input shapes
    if x.ndim != 1 or y.ndim != 1:
        raise Exception("x and y need to be vectors")
    if (z.ndim != 2) or (z.shape[0] != y.shape) != (z.shape[1] != x.shape):
        raise Exception("z needs to be 2 dimensional with size y*x")
    if xp.ndim != 1 or yp.ndim != 1:
        raise Exception("xp and yp need to be vectors")

    # check values of xp and yp:
    if np.min(xp) < x[0] or np.max(xp) > x[-1]:
        raise Exception("xp value outside data range")
    if np.min(yp) < y[0] or np.max(yp) > y[-1]:
        raise Exception("yp value outside data range")

    # bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
    # rewrite the equation to:
    # zi = (1 - yr)*(xr*z[yc, xc+1] + z[yc, xc]*(1-xr)) + yr*(xr*z[yc+1, xc+1]+ z[yc+1, xc]*(1-xr))
    #
    # with:
    # xt, yt: sample points (x, y in Wikipedia)
    # xc, yc: integer part of xt, yt coordinates (x1, y1 in Wikipedia)
    # xr, yr: float part of xt, yt coordinates ((x-x1)/(x2-x1), (y-y1)/(y2-y1) in Wikipedia)

    xt = (xp - x[0]) / (x[1] - x[0])
    yt = (yp - y[0]) / (y[1] - y[0])

    if method == 'linear':
        # this part is faster than using np.divmod
        xc = np.floor(xt).astype(int)
        yc = np.floor(yt).astype(int)
        xr = xt - xc
        yr = yt - yc

        # save multiply used variables for speedup
        a = z[yc, xc]
        b = z[yc+1, xc]

        # rearranged form with only 4 multiplications for speedup
        return (1 - yr) * (xr * (z[yc, xc + 1] - a) + a) + yr * (xr * (z[yc + 1, xc + 1] - b) + b)

    elif method == 'nearest':
        # fast rounding for positive numbers
        xc = (xt + 0.5).astype(int)
        yc = (yt + 0.5).astype(int)

        return z[yc, xc]
    else:
        raise Exception("Invalid method")
