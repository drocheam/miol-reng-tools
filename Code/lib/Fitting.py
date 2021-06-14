import numpy as np

# Author: Damian Mendroch,
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Functions for curve regression and fitting (1D)

Supported:
* Conic Section (fast implementation using pseudoinverse)
* Circle (fast implementation using pseudoinverse)
* Polynomial (implementation using numpy's poly1d)
* Axial Symmetric Polynomial (implementation using numpy's poly1d)

The implementations for Cone Section and Circle uses linear methods for regression,
this is done by writing the fitting functions as a system of linear equation with unknowns, from which the actual
desired unknowns can be determined. The overdetermined equation system is solved using the QR method, this leads to
a least square solution for the paramaters.
See ConicSectionRegression() for an example.

Compared to universal methods like gradient descent, this algortihm is much faster, needs no starting values
and is not susceptible for landing in a local minimum.
"""


def ConicSection(r, z0, p, k):
    """
    calculate the conic section curve (formula according to DIN ISO 10110 with additional offset z0)

    :param r: ordinate values (numpy 1D array)
    :param z0: offset (float)
    :param p: curvature constant (float)
    :param k: conic constant (float)
    :return: abscissa values (1D numpy array)
    """
    return z0 + p*r**2 / (1 + np.sqrt(1 - (1+k)*(p*r)**2))


def ConicSectionDerivative(r, z0, p, k):
    """
    calculate the derivative of a conic section

    :param r: ordinate values (numpy 1D array)
    :param z0: offset (float)
    :param p: curvature constant (float)
    :param k: conic constant (float)
    :return: derivative values (1D numpy array)
    """
    return p*r / np.sqrt(1 - (1+k)*(p*r)**2)


def ConicSectionCircle(r, z0, p, k):
    """
    calculate the inner curvature circle of a conic section (corresponds to the conic sections with k=0)

    :param r: ordinate values (numpy 1D array)
    :param z0: offset (float)
    :param p: curvature constant (float)
    :param k: conic constant (float)
    :return: curvature circle (1D numpy array)
    """
    return ConicSection(r, z0, p, k=0)



def ConicSectionRegression(r, z, cp=1):
    """
    calculate conic section regression
    cp specifies which lower fraction of r and z to use,
    this can be helpful if the curve deviates from a conic section in the outer region

    :param r: ordinate values (1D array)
    :param z: abscissa values (1D array)
    :param cp: cut parameter (float between 0 and 1)
    :return: z0: offset (float), p: curvature constant (float), k: conic constant (float)
    """

    # use only finite data
    mask = np.isfinite(z)

    # exclude last part as specified by cut parameter cp
    mask[round(cp*z.shape[0]):] = False

    # for a 2D surface, including shift in x, y  and z direction (x0, y0, z0):
    # rewrite conic section formula z = z0 + p(x² + y²) / (1 + sqrt(1 - (1+k)p²(x² + y²)))
    # as x² + y² = - z² * (1+k) + 2z * (1/p + z0(1+k)) + 2y*(y0) + 2x(x0) - 1 * (2z0/p + z0²(1+k) + x0² + y0²)
    # b = Ax, with b = x² + y² and A = [-z²; 2z; 2y; 2x; -1].
    #          and x = [(1 + k);  (1/p + z0(1+k));  y0;  x0;  (2z0/p + z0²(1+k) + x0² + y0²)]

    # for a 1D curve with only offset z0:
    # rewrite conic section formula z = z0 + p*r² / (1 + sqrt(1 - (1+k)p²r²))
    # as r² = -z²(1+k) + 2z(1/p + z0(1+k)) - 1*(2z0/p + z0²(1+k))
    # b = Ax, with b = r² and A = [-z²; 2z; -1]. x = [(1 + k);  (1/p + z0(1+k));  (2z0/p + z0²(1+k))]
    A = np.zeros((mask.nonzero()[0].shape[0], 3))
    A[:, 0] = -z[mask]**2
    A[:, 1] = 2*z[mask]
    A[:, 2] = -1
    b = r[mask]**2

    # solve overdetermined equation system using QR method
    Q, R = np.linalg.qr(A)
    X = np.linalg.inv(R) @ Q.T @ b # X = R^-1 * Q^T * b

    # case k = -1 (corresponds to X[1] = 0)
    if np.abs(X[0]) < np.finfo(float).eps * 5:
        # pole X[1] = 0 ignored, because it would equal 1/p = 0 for k = -1 and therefore an infinitesimal body
        return X[2]/X[1]/2, 1/X[1], -1.0 # return z0, p, k

    # typical case
    # pole X[1] - z0 * X[0] = 0 is ignored, because it would equal 1/p = 0 and therefore an infinitesimal body
    # for real numbers p, k the term in the square root is always >= 0
    z01 = X[1]/X[0] - np.sqrt((X[1]/X[0])**2 -X[2]/X[0])
    z02 = X[1]/X[0] + np.sqrt((X[1]/X[0])**2 -X[2]/X[0])
    p1 = 1/(X[1] - z01*X[0])
    p2 = 1/(X[1] - z02*X[0])
    k = X[0] - 1

    # check which possible solution satisfies best
    abs1 = np.mean(np.abs(z[mask] - ConicSection(r[mask], z01, p1, k)))
    abs2 = np.mean(np.abs(z[mask] - ConicSection(r[mask], z02, p2, k)))

    if abs1 < abs2:
        return z01, p1, k
    return z02, p2, k


def Circle(r, z0, R, sign):
    """
    calculate circle curve

    :param r: ordinate values (1D array)
    :param z0: offset (float)
    :param R: radius (float)
    :param sign: curvature, -1 or 1 (int)
    :return: 1D numpy array of abscissa values
    """
    return sign*np.sqrt(R**2 - r**2) + z0


def CircleDerivative(r, z0, R, sign):
    """
    calculate circle derivative curve

    :param r: ordinate values (1D array)
    :param z0: offset (float)
    :param R: radius (float)
    :param sign: curvature, -1 or 1 (int)
    :return: 1D numpy array of abscissa values
    """
    return -sign*r/np.sqrt(R**2 - r**2)


def CircleRegression(r, z, cp=1):
    """
    calculate circle regression
    cp specifies which lower fraction of r and z to use, this can be helpful if the curve deviates
    from a conic section in the outer region

    :param r: ordinate values (1D array)
    :param z: abscissa values (1D array)
    :param cp: cut parameter (float between 0 and 1)
    :return: z0: offset (float), R: radius (float), sign: -1 or 1 (int)
    """

    # use only finite data
    mask = np.isfinite(z)
    # exclude last part as specified by cut parameter cp
    mask[round(cp*z.shape[0]):] = False

    # Circle Formula R² = r² + (z - z0)²
    # Write as r² + z² =  2z(z0) + 1*(R² -  z0²)
    # Can be written in form b = Ax
    A = np.zeros((mask.nonzero()[0].shape[0], 2))
    A[:, 0] = 2 * z[mask]
    A[:, 1] = 1
    b = r[mask] ** 2 + z[mask] ** 2

    # solve overdetermined equation system using the QR Algorithm
    Q, R = np.linalg.qr(A)
    X = np.linalg.inv(R) @ Q.T @ b # X = R^-1 * Q^T * b

    # calculate Parameters
    z0 = X[0]
    R = np.sqrt(X[1] + X[0]**2)

    # check which possible solution satisfies best
    abs1 = np.mean(np.abs(z[mask] - Circle(r[mask], z0, R,  1)))
    abs2 = np.mean(np.abs(z[mask] - Circle(r[mask], z0, R, -1)))

    if abs1 < abs2:
        return z0, R, 1
    return z0, R, -1


def Poly(x, a):
    """
    calculate polynomial values from coefficients

    :param x: ordinate values (1D array)
    :param a: coefficients from np.poly1d (1D array)
    :return: 1D numpy array of polynomial values
    """
    poly = np.poly1d(a)
    return poly(x)


def PolyDerivative(x, a):
    """
    calculate derivative of polynomial from coefficients

    :param x: ordinate values (1D array)
    :param a: oefficients from np.poly1d (1D array)
    :return: 1D numpy array of derivative values
    """
    der = np.polyder(np.poly1d(a))
    return der(x)


def PolyRegression(x, y, order=8):
    """
    calculate polynomial regression for specified order

    :param x: ordinate values (1D array)
    :param y: abscissa values (1D array
    :param order: polynomial order (int)
    :return: coefficients in reverse order (1D numpy array)
    """
    return np.polyfit(x, y, order)


def SymPolyRegression(x, y, order=8):
    """
    calculate regression for even-only polynomials

    :param x: ordinate values (1D array)
    :param y: abscissa values (1D array)
    :param order: polynomial order (int)
    :return: coefficients in reverse order with odd ones being zero (1D numpy array)
    """
    # add mirrored data to enforce symmetrical polynomial fitting
    mask = x > 0 # exclude x = 0 in mask, because it would arise twice in x_s
    x_s = np.concatenate((-np.flip(x[mask]), x[mask]))
    y_s = np.concatenate((np.flip(y[mask]), y[mask]))

    s = np.polyfit(x_s, y_s, order) # calculate polyfit
    s[-2::-2] = 0 # set residual asymmetrical polynomial components to zero

    return s
