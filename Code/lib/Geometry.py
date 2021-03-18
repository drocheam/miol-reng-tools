from lib.Interpolation import interp2f
from lib.Fitting import *
import numpy as np
from scipy import interpolate


"""
Methods for geometrical manipulation and information on the lens

cutting to a circle: cutLens()
detection of a small tilt of the lens: tiltRegression()
generate radial profiles from lens images and correct tilt: getProfiles()
information on asphere fitting (cone section and polynomial): ProfileInformation()
"""


def cutLens(x, y, h_data, R, xm=0, ym=0):
    """
    sets all data outside a circle defined by xm, ym, R to nan

    :param x: x values (1D array)
    :param y: y values (1D array)
    :param h_data: z values (2D array)
    :param R: circle radius (float)
    :param xm: circle centre x coordinate (float)
    :param ym: circle centre y coordinate (float
    :return: z value 2D array
    """

    X, Y = np.meshgrid(x, y)
    outside = (X-xm)**2 + (Y-ym)**2 > (R + x[1]-x[0])**2

    h_data_c = h_data.copy()
    h_data_c[outside] = np.nan

    return h_data_c



def tiltRegression(x, y, h_data, S, cp=1, tr=0.09):
    """
    regression for the x and y tilt of the lens with axial symmetry in x and y direction
    around a point specified in dictionary S. The tilt needs to be rather small for the algorithm
    to work correctly (<3deg), see the documentation for more details

    :param x: x values (1D array)
    :param y: y values (1D array
    :param h_data: z values (2D array)
    :param S: lens property dictionary
    :param cp: cut parameter setting (float from 0 to 1, and -1 for automatic mode)
    :param tr: threshold for cp calculation (float)
    :return: x direction tilt (float), y direction tilt (float)
    """

    # calculate nearest point to symmetry centre
    ds = x[2]-x[1]
    ymi = round((S['ym']-y[0])/ds)
    xmi = round((S['xm']-x[0])/ds)

    # we assume, that tilt only changed the z values, but not x and y
    # this simplification holds true for small angles and flat objects
    # using the available data we calculate a cut parameter (cp),
    # that defines a region from the midpoint, where this is true

    # automatic cut detection
    if cp == -1:
        # minimum cp parameter
        cp_min = 0.4
        # array slice from inner circle (spec. by cp_min) to lens edge:
        ring = np.arange(cp_min*S['r2']/ds, S['r2']/ds).astype(int)

        # neglect region if abs(z-z0)*sin(a)/r > tol  ->  abs(z-z0) > tr*r
        neglect = np.abs(h_data[ymi, xmi+ring] - h_data[ymi, xmi]) > tr*ring*ds
        pos = np.where(neglect)[0] # find neglected positions

        # set cp to first neglected position (if existing) else to 1
        cp = cp_min + pos[0]*ds/S['r2'] if len(pos) > 0 else 1

    # manual cut detection = cp specified by parameter

    # cut lens to region
    if cp < 1:
        h_data = cutLens(x, y, h_data.copy(), cp*S['r2'], S['xm'], S['ym'])

    # generate opposing side matrices (x and y symmetry line with origin at (xm, ym))
    X1 = np.fliplr(h_data[:, :xmi]) # data left of y symmetry axis
    X2 = h_data[:, xmi:] # data right of y symmetry axis
    Y1 = np.fliplr(h_data[:ymi, :].T) # data above of x symmetry axis
    Y2 = h_data[ymi:, :].T # data below of x symmetry axis

    # size of overlap of the matrices
    sizeX = min(X1.shape[1], X2.shape[1])
    sizeY = min(Y1.shape[1], Y2.shape[1])

    # subtract opposite sides, divide difference by 2
    dhX = 0.5 * (X2[:, :sizeX] - X1[:, :sizeX])
    dhY = 0.5 * (Y2[:, :sizeY] - Y1[:, :sizeY])

    # estimate slope for all x slices
    xn = np.arange(sizeX)*ds # x vector
    mX = np.ones(dhX.shape[0]) * np.nan # initialize result with nan
    for n in np.arange(dhX.shape[0]):
        mask = np.isfinite(dhX[n,:]) # use only finite data
        if np.count_nonzero(mask) > 10: #if there are enough elements in the mask
            mX[n] = np.mean(dhX[n, mask]) / np.mean(xn[mask]) # estimate slope

    # do the same for y slices
    yn = np.arange(sizeY)*ds
    mY = np.ones(dhY.shape[0]) * np.nan
    for n in np.arange(dhY.shape[0]):
        mask = np.isfinite(dhY[n,:])
        if np.count_nonzero(mask)  > 10:
            mY[n] = np.mean(dhY[n, mask]) / np.mean(yn[mask])

    # calculate angles from slope
    radY = -np.arcsin(np.nanmean(mX)) # tilt in y direction
    radX =  np.arcsin(np.nanmean(mY)/np.cos(radY)) # tilt in x direction

    return radX, radY


def getProfiles(x_in, y_in, h_data, S, rad=[]):
    """
    generates profiles from from lens centre to outer edge.
    If rad is specified a tilt correction is applied

    :param x_in: x values (1D array)
    :param y_in: y values (1D array)
    :param h_data: z values (2D array)
    :param S: lens property dictionary
    :param rad: (optional): tilt values in x and y direction (list of 2 floats)
    :return: r vector (1D array), left profiles and right profiles (both 2D arrays, with profiles in rows)
    """

    # transform coordinates so centre lies at (0,0)
    x, y = x_in - S['xm'], y_in - S['ym']

    theta = np.arcsin(y / S['r2']) # profile angles
    r = np.arange(0, S['r2'], x[1]-x[0]) # r vector

    # Generate coordinates for all profile sample points
    Rqq, cosT = np.meshgrid(r, np.cos(theta))
    Rqq, sinT = np.meshgrid(r, np.sin(theta))
    Xqq, Yqq = Rqq * cosT, Rqq * sinT
    xp = np.concatenate((-Xqq.ravel(), Xqq.ravel())) # x coordinates for all sample points
    yp = np.concatenate((Yqq.ravel(), Yqq.ravel())) # y coordinates for all sample points

    # use fast bilinear Interpolation
    prof = interp2f(x, y, h_data, xp, yp, 'linear')

    # reshape
    prof0 = prof.reshape(y.shape[0]*2, r.shape[0]) # reshape so profiles are in rows
    prof1 = prof0[:y.shape[0], :] # left profiles
    prof2 = prof0[y.shape[0]:, :] # right profiles

    # tilt lens
    if len(rad) != 0:
        for n in np.arange(theta.shape[0]):
            # due to rotation r and z change to
            # rs = sqrt( (x*cos(beta)+(z-z0)*sin(beta))^2  + (y*cos(alpha)-(z-z0)*sin(alpha))^2 )
            # zs = z - x*sin(beta) + y*sin(alpha)

            alpha, beta = rad
            x = r*np.cos(theta[n])
            y = r*np.sin(theta[n])
            dz1 = prof1[n, :]-prof1[n, 0] # (z-z0) profiles left
            dz2 = prof2[n, :]-prof2[n, 0] # (z-z0) profiles right

            # interpolate sample values, note that x is negative so it deviates from above formula
            rs = np.hypot(-x*np.cos(beta) + dz1*np.sin(beta), y*np.cos(alpha) - dz1*np.sin(alpha))
            prof1[n, :] += x*np.sin(beta) + y*np.sin(alpha) # zs
            f1 = interpolate.interp1d(rs, prof1[n,:], bounds_error=False) # bounds_error = False sets extrapolated values to nan
            prof1[n,:] = f1(r)

            # same for right profiles
            rs = np.hypot(x*np.cos(beta) + dz2*np.sin(beta), y*np.cos(alpha) - dz2*np.sin(alpha))
            prof2[n, :] += -x*np.sin(beta) + y*np.sin(alpha)
            f2 = interpolate.interp1d(rs, prof2[n,:], bounds_error=False)
            prof2[n,:] = f2(r)

        # data has been stretched/skewed, therefore shorten profiles to valid data
        invalid = np.where(~np.all(np.isfinite(prof1) & np.isfinite(prof2), axis=0))[0]
        if len(invalid):
            return r[:invalid[0]], prof1[:, :invalid[0]], prof2[:, :invalid[0]]

    return r, prof1, prof2



def ProfileInformation(CR, PR, r):
    """
    asphere profile information shows regression parameters in a readable way
    note that only even polynomial coefficients are shown

    :param CR: conic section regression results fom ConicSectionRegression() (1D array)
    :param PR: polynomial regression results fom PolyRegression() (1D array)
    """

    if CR[2] >= 0:      kind = "oblate ellipse"
    elif CR[2] <= -1:   kind = "hyperbola"
    else:               kind = "prolate ellipse"

    if np.abs(CR[2]) < 0.1:
        kind += ", close to a circle (k = 0)"
    elif np.abs(CR[2] + 1) < 0.1:
        kind += ", close to a parabola (k = -1)"

    print("\nProfile Information")
    print("Conic Constant k =", "{:.4f}".format(CR[2]), "(", kind, ")")
    print("Curvature Radius R =", "{:.3f}".format(1 / CR[1] / 1000), "mm")

    print("\nPolynomial Coefficients (a0, a2, a4, ...):")
    print(PR[::-2])

    # q denotes the change of the height from polynomials and conic section as ratio
    p = Poly(r, PR)
    c = ConicSection(r, *CR)
    q = (np.max(p) - np.min(p)) / (np.max(c) - np.min(c))
    print("Height change ratio q = ", "{:.4f}".format(q))

    print("\nOptical diameter d_o = ", 2*r[-1]/1000, "mm")