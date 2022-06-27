from lib.Interpolation import interp2f, interpolateNan
from lib.Fitting import *
import numpy as np
from scipy import interpolate
import scipy.optimize
from scipy.ndimage import gaussian_filter as gaussFilter

# Author: Damian Mendroch
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Methods for geometrical manipulation and information on the lens

cutting to a circle: cutLens()
detection of a small tilt of the lens: tiltRegression()
generate radial profiles from lens images and correct tilt: getProfiles()
information on asphere fitting (cone section and polynomial): ProfileInformation()
"""


def cutLens(x: np.ndarray, y: np.ndarray, h_data: np.ndarray, R: float, xm: float=0, ym: float=0) -> np.ndarray:
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


def getProfiles(x_in: np.ndarray, y_in: np.ndarray, h_data: np.ndarray, S: dict, rad: list=[])\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    theta = np.arcsin(y / S['r2'])  # profile angles
    r = np.arange(0, S['r2'], x[1]-x[0])  # r vector

    # Generate coordinates for all profile sample points
    Rqq, cosT = np.meshgrid(r, np.cos(theta))
    Rqq, sinT = np.meshgrid(r, np.sin(theta))
    Xqq, Yqq = Rqq * cosT, Rqq * sinT
    xp = np.concatenate((-Xqq.ravel(), Xqq.ravel()))  # x coordinates for all sample points
    yp = np.concatenate((Yqq.ravel(), Yqq.ravel()))  # y coordinates for all sample points

    # use fast bilinear Interpolation
    prof = interp2f(x, y, h_data, xp, yp)

    # reshape
    prof0 = prof.reshape(y.shape[0]*2, r.shape[0])  # reshape so profiles are in rows
    prof1 = prof0[:y.shape[0], :]  # left profiles
    prof2 = prof0[y.shape[0]:, :]  # right profiles

    # tilt lens
    if len(rad) != 0:

        b, a = rad[0], rad[1]
        veca = np.array([0, 0, 1])
        vecb = np.array([np.sin(b), -np.sin(a), np.sqrt(1 - np.sin(a) ** 2 - np.sin(b) ** 2)])

        # variables for rotation matrix calculation
        v = np.cross(veca, vecb)
        c = np.dot(veca, vecb)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        # equation source: https://math.stackexchange.com/a/476311
        # calculate rotation matrix for veca -> vecb
        R = np.eye(3) + vx + vx @ vx / (1 + c)

        dz1 = prof1 - prof1[:, 0, np.newaxis]  # (z-z0) profiles left
        dz2 = prof2 - prof2[:, 0, np.newaxis]  # (z-z0) profiles right

        for n in np.arange(theta.shape[0]):

            x = r*np.cos(theta[n])
            y = r*np.sin(theta[n])

            # rotated x, y, z
            xyz = R @ np.row_stack((x, y, dz1[n]))
            x1, y1, z1 = xyz[0, :], xyz[1, :], xyz[2, :]

            rs = np.sqrt(x1**2 + y1**2)  # actual r vector
            prof1[n, :] = prof1[n, 0] + z1  # actual height values
            f1 = interpolate.interp1d(rs, prof1[n,:], bounds_error=False)
            # bounds_error = False sets extrapolated values to nan
            prof1[n,:] = f1(r)  # interpolate values on r grid from before

            xyz = R @ np.row_stack((-x, y, dz2[n]))
            x2, y2, z2 = xyz[0, :], xyz[1, :], xyz[2, :]

            rs = np.sqrt(x2**2 + y2**2)
            prof2[n, :] = prof2[n, 0] + z2
            f2 = interpolate.interp1d(rs, prof2[n,:], bounds_error=False)
            prof2[n,:] = f2(r)

        # data has been stretched/skewed, therefore shorten profiles to valid data
        invalid = np.where(~np.all(np.isfinite(prof1) & np.isfinite(prof2), axis=0))[0]
        if len(invalid):
            return r[:invalid[0]], prof1[:, :invalid[0]], prof2[:, :invalid[0]]

    return r, prof1, prof2


def removeOutliers(h_data, fc, tr):
    """
    2D surface data outlier removal. Compares original and filtered data (gaussian 2D filtering),
    if the difference is above 'tr', the data point is set to nan.

    :param h_data: z value array (2D numpy array)
    :param fc: spatial filtering constant, gaussian filtering variance in data points (positive float)
    :param tr: outlier threshold (float)
    :return: h_data with outliers removed (2D numpy array)
    """
    h_data_k = h_data.copy()

    # use interpolated data, so we have data points for filtering there for filtering in next step
    h_data_i = interpolateNan(h_data_k)

    # on nearly steady slopes the 'nearest' mode is suited better compared of default 'mirror'
    h_data_f = gaussFilter(h_data_i, fc, mode='nearest')

    # if any of the data points currently processed by a filtering kernel has a nan value, the result is also nan.
    # The regions with finite data shrink as a result. To counter this, detect this new introduced nans and
    # replace them with the original data. In doing so, no outlier detection can be made in these regions.
    mask = np.isfinite(h_data) & ~np.isfinite(h_data_f)
    h_data_f[mask] = h_data[mask]

    # set outlying data to nan
    dev = np.abs(h_data_f - h_data_i)
    h_data_k[dev > tr] = np.nan

    return h_data_k


def ProfileInformation(CR: np.ndarray, PR: np.ndarray, r: np.ndarray, h: float) -> None:
    """
    asphere profile information shows regression parameters in a readable way
    note that only even polynomial coefficients are shown

    :param CR: conic section regression results fom ConicSectionRegression() (1D array)
    :param PR: polynomial regression results fom PolyRegression() (1D array)
    :param r: radial vector (1D numpy array)
    :param h: height of profile
    """

    if CR[2] >= 0:      kind = "oblate ellipse"
    elif CR[2] <= -1:   kind = "hyperbola"
    else:               kind = "prolate ellipse"

    if np.abs(CR[2]) < 0.1:
        kind += ", close to a circle (k = 0)"
    elif np.abs(CR[2] + 1) < 0.1:
        kind += ", close to a parabola (k = -1)"

    print("\nProfile Information")
    print(f"Conic Constant k = {CR[2]:.4f} ({kind})")
    print(f"Curvature Radius R = {1/CR[1]/1000:.3f}mm")

    print("\nPolynomial Coefficients (a0, a2, a4, ...):")
    print(PR[::-2])

    # q denotes the change of the height from polynomials and conic section as ratio
    p = Poly(r, PR)
    c = ConicSection(r, *CR)
    min_, max_ = np.min(p), np.max(p)
    h_p = max_ if np.abs(max_) > np.abs(min_) else min_
    # h_p = (np.max(p) - np.min(p)) / (np.max(c) - np.min(c))

    print(f"Height h = {h:.4f}µm")
    print(f"Height change due to polynomial h_p = {h_p:.4f}µm")
    print(f"\nOptical diameter d_o = {2*r[-1]/1000}mm")


def shrinkData(x, y, Z, factor):
    """
    Reduces number of elements of data by a factor.
    Applies gaussian filtering before interpolation.

    :param x: x values (numpy 1D array)
    :param y: y values (numpy 1D array)
    :param Z: z values (numpy 2D array)
    :param factor: overall size downscale factor
    :return: downscaled x, y, Z
    """
    k = np.sqrt(factor)
    ds = x[1]-x[0]

    # 2D gaussian filtering
    Zf = gaussFilter(Z, k)

    # filtering grows nan regions, add unfiltered data back where filtering introduced new nans
    mask = np.isfinite(Z) & ~np.isfinite(Zf)
    Zf[mask] = Z[mask]

    # create shrunk grid and interpolate data on this grid
    xi = np.arange(x[0], x[-1], ds*k)
    yi = np.arange(y[0], y[-1], ds*k)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interp2f(x, y, Zf, Xi.ravel(), Yi.ravel())
    Zi = zi.reshape(Xi.shape)

    return xi, yi, Zi


def getTiltCorrection(xi: np.ndarray, yi: np.ndarray, h_data: np.ndarray, LP: dict, cut: float) -> list[float, float]:
    """
    Get values for tilt detection. 
    Uses optimization functions to determine a best fit tilt vector using properties of rotational symmetry

    :param xi: x values (numpy 1D array)
    :param yi: y values (numpy 1D array)
    :param h_data: z values (numpy 2D array)
    :param LP: lens property dict
    :param cut: cut parameter, scales lens radius, everything above will be excluded for tilt determination
    :return: tilt around y-axis, tilt around x-axis
    """
    x, y, h_data_c = shrinkData(xi, yi, h_data, 3)
    h_data_c = cutLens(x, y, h_data_c, cut*LP['r2'], LP['xm'], LP['ym'])

    def optY(args, ang):
        r, prof1, prof2 = getProfiles(x, y, h_data_c, LP, [ang, args[0]])
        return 1000*np.mean(np.std(prof1, axis=0)**2+np.std(prof2, axis=0)**2)

    def optX(args, ang):
        r, prof1, prof2 = getProfiles(x, y, h_data_c, LP, [args[0], ang])
        return 1000*np.mean((np.mean(prof1, axis=0)-np.mean(prof2, axis=0))**2)
    
    dt = 4/360*2*np.pi
    bounds = ((-dt, dt),)

    res = scipy.optimize.minimize(optX, (0,), args=(0), bounds=bounds, method="Powell")
    res2 = scipy.optimize.minimize(optY, (0,), args=(res.x[0]), bounds=bounds, method="Powell")
    
    return [res.x[0], res2.x[0]]

