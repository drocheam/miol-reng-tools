import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PyQt5 # for plotting backend
from lib.Interpolation import interp2f

"""
Plotting functions used throughout the library and scripts

"""

matplotlib.use('Qt5Agg')

# fonts to make everything look more professional
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['figure.raise_window'] = True

# figure size
fsize = (9, 7)


# parameters for paper output
#matplotlib.rcParams["figure.subplot.bottom"] = 0.157
#matplotlib.rcParams["figure.subplot.top"] = 0.91
#matplotlib.rcParams["figure.subplot.right"] = 0.97
#matplotlib.rcParams["figure.subplot.left"] = 0.112
#fsize = (6, 2.8)


# parameters for poster output
#matplotlib.rcParams["figure.subplot.bottom"] = 0.19
#matplotlib.rcParams["figure.subplot.top"] = 0.91
#matplotlib.rcParams["figure.subplot.right"] = 0.97
#matplotlib.rcParams["figure.subplot.left"] = 0.112
#fsize = (8, 2.25)
#matplotlib.rcParams["legend.loc"] = 'upper right'


def showTicks():
    """
    shows major and minor ticks for current plot, with the minor ones being more subtle
    """
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()


def LensPlot(x, y, h_data_in, blocking=True):
    """
    Lens plot, featuring 2D height color map and contour plot and a x-z plot for a set y value

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data_in: z values (2D array)
    :param blocking: if the pyplot plot pauses further program execution (bool)
    """

    h_data = np.flipud(h_data_in) # flip so (0,0) is in lower left corner
    ext = [x[0], x[-1], y[0], y[-1]]

    plt.figure(figsize=fsize)
    plt.subplot(211)
    plt.imshow(h_data, extent=ext)
    cbar = plt.colorbar(orientation='horizontal', shrink = 0.3)
    plt.contour(h_data_in, levels=8, extent=ext, colors='red')
    cbar.ax.set_xlabel('h in µm')
    plt.xlabel("x in µm")
    plt.ylabel("y in µm")
    plt.title("Height Colormap and Contours")

    plt.subplot(212)
    mid = h_data.shape[0] // 2
    plt.plot(x, h_data[mid, :])
    plt.xlim(x[0], x[-1])
    showTicks()
    plt.xlabel("x in µm")
    plt.ylabel("h in µm")
    plt.title("Height Profile for y = " + str(y[mid]) + "µm")

    plt.show(block=blocking)
    plt.pause(0.1)


def DiffLensPlot(x, y, h_data_in, S=dict(), blocking=True):
    """
    Two plots of the normalized first and second gradient
    optional: shows specified lens properties (lens axis, and edge) in plots

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data_in: z values (2D array)
    :param S: (optional): lens property struct: xm, ym Coordinates of Lens Axis, r1 Radius of Aid Circle, r2 Lens Radius
    :param blocking: if the pyplot plot pauses further program execution (bool)
    """

    h_data = np.flipud(h_data_in) # flip so (0,0) is in lower left corner
    ext = [x[0], x[-1], y[0], y[-1]]

    # simple absolute gradient emphasizes sharp edges while small change are barely visible
    # -> use nonlinear sigmoid function (e.g. x/sqrt(1 + x^2)) to make small changes visible (range compression)
    # normalize gradient xg at a certain quantile x0 -> x/sqrt(1 + x^2) with x = xg/x0: xg/sqrt(x0^2 + xg^2))

    # first gradient
    grad1v = np.gradient(h_data)
    grad1 = np.hypot(grad1v[0], grad1v[1])
    grad1m = np.nanquantile(grad1, 0.65)
    grad1n = grad1/np.hypot(grad1, grad1m)

    # second gradient
    grad2v = np.gradient(grad1)
    grad2 = np.hypot(grad2v[0], grad2v[1])
    grad2m = np.nanquantile(grad2, 0.65)
    grad2n = grad2/np.hypot(grad2, grad2m)

    plt.figure(figsize=fsize)

    plt.subplot(211)
    plt.imshow(grad1n,  extent=ext)
    if len(S) > 0: # if S dictionary given
        circ1 = plt.Circle((S['xm'], S['ym']), S['r1'], fill=False, edgecolor='r')
        circ2 = plt.Circle((S['xm'], S['ym']), S['r2'], fill=False, edgecolor='r')
        plt.scatter(S['xm'], S['ym'], color='r', marker='x', s=50)
        plt.gca().add_artist(circ1)
        plt.gca().add_artist(circ2)
    plt.xlabel("x in µm")
    plt.ylabel("y in µm")
    plt.title("Compressed Gradient")

    plt.subplot(212)
    plt.imshow(grad2n,  extent=ext)
    if len(S) > 0:
        circ1 = plt.Circle((S['xm'], S['ym']), S['r1'], fill=False, edgecolor='r')
        circ2 = plt.Circle((S['xm'], S['ym']), S['r2'], fill=False, edgecolor='r')
        plt.scatter(S['xm'], S['ym'], color='r', marker='x', s=50)
        plt.gca().add_artist(circ1)
        plt.gca().add_artist(circ2)
    plt.xlabel("x in µm")
    plt.ylabel("y in µm")
    plt.title("Compressed Second Gradient")

    plt.show(block=blocking)
    plt.pause(0.1)


def ComparisonPlot(x, y1, y2, blocking=True, title="Height Profile Variation"):
    """
    shows a comparison of two profile datasets
    the variation is shown (area between min and max data in dataset) as well as the mean of both datasets

    :param x: ordinate vector for both profiles (1D array)
    :param y1: profile dataset 1 with the profiles in rows (2D array)
    :param y2: profile dataset 2 with the profiles in rows (2D array)
    :param blocking: if the pyplot plot pauses further program execution (bool, defaults to true)
    :param title: title for the plot (string)
    """

    y1ma = np.nanmax(y1, axis=0)
    y1mi = np.nanmin(y1, axis=0)
    y1me = np.nanmean(y1, axis=0)

    y2ma = np.nanmin(y2, axis=0)
    y2mi = np.nanmax(y2, axis=0)
    y2me = np.nanmean(y2, axis=0)

    plt.figure(figsize=fsize)

    plt.fill_between(x, y1mi, y1ma, alpha=0.3)
    plt.plot(x, y1me)
    plt.fill_between(x, y2mi, y2ma, alpha=0.3)
    plt.plot(x, y2me)
    #plt.plot(x, (y1me+ y2me)/2)

    showTicks()
    plt.xlim(x[0], x[-1])

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    plt.title(title)
    #plt.legend(["Profile 1 mean", "Profile 2 mean", "Profile mean", "Profile 1 Variation", "Profile 2 Variation"])
    plt.legend(["Profile 1 mean", "Profile 2 mean", "Profile 1 Variation", "Profile 2 Variation"])

    plt.show(block=blocking)
    plt.pause(0.1)


def ProfilePlot(r, profs, legentries=[], title="Height Profile", blocking=True):
    """
    shows one or multiple profiles in a plot

    :param r: ordinate vector for both profiles (1D array)
    :param profs: profiles, either as tuple of 1D arrays (for multiple profiles) or as one 1D array (for one profile)
    :param legentries: legend entry list for legend() function (list of strings)
    :param title: title for the plot (string, has a default value
    :param blocking: if the pyplot plot pauses further program execution (bool, defaults to true)
    """

    plt.figure(figsize=fsize)

    # set plot and legend depending on input type
    if type(profs) == tuple:
        for n in np.arange(len(profs)):
            plt.plot(r, profs[n])

        if not legentries:
            for n in np.arange(len(profs)):
                legentries.append("Profile " + str(n+1))
    else:
        plt.plot(r, profs)
        if not legentries:
            legentries.append("Profile")

    showTicks()
    plt.xlim(r[0], r[-1])

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    plt.title(title)
    plt.legend(legentries)

    plt.show(block=blocking)
    plt.pause(0.1)


def ThicknessPlot(x, y, h_data_in, T, blocking=True):
    """
    Overlays two edge lines to a 2D height color map plot.
    Thickness is measured using the difference between those points including a height difference of the profile.
    (For this the assumption is that the edge is parallel to the optical axis).

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data_in: z values (2D array)
    :param T: (optional): lines adjustment 1D array, containing coordinates of a point A and B opposite
                of each other at the lens edge: [Ax, Ay, Bx, By]
    :param blocking: if pyplot video pauses further program execution (bool)
    :return: edge thickness (numpy float)
    """

    h_data = np.flipud(h_data_in) # flip so (0,0) is in the lower left corner

    # change plot size and subplots depending on point coordinates being specified
    if len(T) > 0:
        plt.figure(figsize=(fsize[0]*1.3, fsize[1]))
        plt.subplot(121)
    else:
        plt.figure(figsize=fsize)

    plt.imshow(h_data, extent=[x[0], x[-1], y[0], y[-1]], cmap='summer')
    plt.xlabel("x in µm")
    plt.ylabel("y in µm")
    cbar = plt.colorbar(orientation='horizontal', shrink = 0.3)
    cbar.ax.set_xlabel('h in µm')
    plt.title("Lens Edge")

    if len(T) > 0:
        # diameter line coordinates between point A and B
        Ax, Ay, Bx, By = T
        if Ax > Bx:
            Ax, Ay, Bx, By =  Bx, By, Ax, Ay # swap so point A is always left

        m = (By-Ay)/(Bx-Ax) # slope diameter line
        m2 = -1/m # slope edge lines (=perpendicular to diameter line)
        f00, f0e = m2*(x[[0,-1]]-Ax) + Ay # lower edge lines start and end
        f10, f1e = m2*(x[[0,-1]]-Bx) + By # upper edge lines start and end

        plt.plot([Ax, Bx], [Ay, By], color='r') # diameter line
        plt.plot([x[0], x[-1]], [f00, f0e], color='k') # lower edge line
        plt.plot([x[0], x[-1]], [f10, f1e], color='k') # upper edge line

        plt.scatter(Ax, Ay, color='r', marker='x') # left diameter point
        plt.scatter(Bx, By, color='r', marker='x') # right diameter point

        plt.legend(["thickness line", "edge lines"])

        # adapt axes
        plt.ylim(y[0], y[-1])
        plt.xlim(x[0], x[-1])

        # points on diameter line
        r = np.arange(0, np.hypot(Bx-Ax, By-Ay), x[1]-x[0])
        x_z = r*np.cos(np.arctan(m)) + Ax
        y_z = r*np.sin(np.arctan(m)) + Ay

        # profile at diameter line
        z_line = interp2f(x, y, np.flipud(h_data), x_z, y_z)

        # mean of end and beginning of diameter line profile
        z1 = np.mean(z_line[np.isfinite(z_line)][:5])
        z2 = np.mean(z_line[np.isfinite(z_line)][-5:])
        r1 = np.mean(r[np.isfinite(z_line)][:5])
        r2 = np.mean(r[np.isfinite(z_line)][-5:])

        d = r[-1] * np.hypot(z2-z1, r2-r1) / (r2-r1) # thickness is scaled by pythagorean factor
        za, ze = (z2-z1)/(r2-r1)*(r[[0, -1]]-r1)  + z1 # z values at beginning and ending of thickness line for plotting

        plt.subplot(122)
        plt.plot(r, z_line, color='g')
        plt.plot([r[0], r[-1]], [za, ze], color='r')
        plt.text((r2+r1)/2, (z1+z2)/2, "thickness d ≈ " + str(np.round(d).astype(int)) + "µm") # text label
        plt.xlabel("r in µm")
        plt.ylabel("h in µm")
        plt.legend(["Edge Profile", "Thickness Line"])
        plt.title("Edge Profile")
        showTicks()

    else: # d not specified
        d = np.nan

    plt.show(block=blocking)
    plt.pause(0.1)

    return d


def interpolationPlot(r, org, interpolated, blocking=True):
    """
    compare original and interpolated profile curve

    :param r: r vector (1D array)
    :param org: original profile (1D array)
    :param interpolated: interpolated profile (1D array)
    :param blocking: if the pyplot plot pauses further program execution (bool, defaults to true)
    """

    plt.figure(figsize=(9, 7))
    plt.plot(r, org, r, interpolated)

    plt.xlim(r[0], r[-1])
    showTicks()
    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    plt.title("Interpolated Height Profile")
    plt.legend(["Original", "Filtered"])

    plt.show(block=blocking)
    plt.pause(0.1)



def FilteredProfilePlot(r, org, filtered, diff2, diff2_filtered, F, blocking=True):
    """
    show filtered profile compared to orginal data in one plot,
    abs of second derivative filtered and unfiltered and threshold in second plot

    :param r: r vector (1D array)
    :param org: original profile (1D array)
    :param filtered: filtered profile (1D array)
    :param diff2: unfiltered abs of second derivative (both 1D arrays)
    :param diff2_filtered: filtered abs of second derivative (both 1D arrays)
    :param F: filter property dictionary, see filterProfile() for more details
    :param blocking: if the pyplot plot pauses further program execution (bool, defaults to true)
    """

    # original and filter plot
    plt.figure(figsize=fsize)
    plt.subplot(211)
    plt.plot(r, org, r, filtered)
    plt.xlim(r[0], r[-1])
    showTicks()

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    plt.title("Height Profile Filtering")
    plt.legend(["original", "filtered"])

    # abs second derivative plot
    plt.subplot(212)
    plt.plot(r, diff2, r, diff2_filtered)
    plt.plot([r[0], r[-1]], [F['tr'], F['tr']])

    showTicks()
    plt.xlim(r[0], r[-1])

    plt.xlabel("r in µm")
    plt.ylabel("|h\'\'| in 1/µm")
    plt.title("Absolute value of second derivative")
    plt.legend(["prefiltered", "pre- and postfiltered", "threshold"])

    plt.show(block=blocking)
    plt.pause(0.1)
