import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PyQt5 # for plotting backend Qt5Agg

# Author: Damian Mendroch
# Project repository: https://github.com/drocheam/miol-reng-tools


"""
Plotting functions
"""

# enforce plotting backend to show plots interactive and in separate windows
matplotlib.use('Qt5Agg')

# better fonts to make everything look more professional
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# raise window to front when showing plot
matplotlib.rcParams['figure.raise_window'] = True

# default output
fsize = (9, 7)
no_title = False

# paper output size
# mm_to_in = 0.0393701
# width_mm = 150
# aspect = 2.25
# fsize = tuple(np.array([width_mm, width_mm/aspect])*mm_to_in)
# no_title = True # figure titles not wanted in journal


def showTicks() -> None:
    """
    shows major and minor ticks for current plot, with the minor ones being more subtle
    """
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
    plt.minorticks_on()


def LensPlot(x: np.ndarray, y: np.ndarray, h_data_in: np.ndarray, blocking: bool=True) -> None:
    """
    Lens plot, featuring 2D height color map and contour plot and a x-z plot for a set y value

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data_in: z values (2D array)
    :param blocking: if pyplot pauses further program execution (bool)
    """

    h_data = np.flipud(h_data_in) # flip so (0,0) is in lower left corner
    ext = [x[0], x[-1], y[0], y[-1]]

    plt.figure(figsize=fsize)
    plt.subplot(211)
    plt.imshow(h_data, extent=ext)
    cbar = plt.colorbar(orientation='horizontal', shrink=0.3)
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
    if not no_title:
        plt.title(f"Height Profile for y = {y[mid]}µm")
    
    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)


def DiffLensPlot(x: np.ndarray, y: np.ndarray, h_data_in: np.ndarray, S: dict=dict(), blocking: bool=True) -> None:
    """
    Two plots of the normalized first and second gradient
    optional: shows specified lens properties (lens axis, and edge) in plots

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data_in: z values (2D array)
    :param S: (optional): lens property struct: xm, ym Coordinates of Lens Axis, r1 Radius of Aid Circle, r2 Lens Radius
    :param blocking: if pyplot pauses further program execution (bool)
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
    if len(S) > 0:  # if S dictionary given
        circ1 = plt.Circle((S['xm'], S['ym']), S['r1'], fill=False, edgecolor='r')
        circ2 = plt.Circle((S['xm'], S['ym']), S['r2'], fill=False, edgecolor='r')
        plt.scatter(S['xm'], S['ym'], color='r', marker='x', s=50)
        plt.gca().add_artist(circ1)
        plt.gca().add_artist(circ2)
    plt.xlabel("x in µm")
    plt.ylabel("y in µm")
    if not no_title:
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
    if not no_title:
        plt.title("Compressed Second Gradient")

    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)


def ComparisonPlot(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, blocking: bool=True,
                   title: str="Height Profile Variation") -> None:
    """
    shows a comparison of two profile datasets
    the variation is shown (area between min and max data in dataset) as well as the mean of both datasets

    :param x: ordinate vector for both profiles (1D array)
    :param y1: profile dataset 1 with the profiles in rows (2D array)
    :param y2: profile dataset 2 with the profiles in rows (2D array)
    :param blocking: if pyplot pauses further program execution (bool, defaults to true)
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

    showTicks()
    plt.xlim(x[0], x[-1])

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    if not no_title:
        plt.title(title)
    plt.legend(["Profile 1 mean", "Profile 2 mean", "Profile 1 Variation", "Profile 2 Variation"])

    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)


def ProfilePlot(r: np.ndarray, profs: np.ndarray, legentries: list=[], 
                title: str="Height Profile", blocking: bool=True) -> None:
    """
    shows one or multiple profiles in a plot

    :param r: ordinate vector for both profiles (1D array)
    :param profs: profiles, either as tuple of 1D arrays (for multiple profiles) or as one 1D array (for one profile)
    :param legentries: legend entry list for legend() function (list of strings)
    :param title: title for the plot (string, has a default value
    :param blocking: if pyplot pauses further program execution (bool, defaults to true)
    """

    plt.figure(figsize=fsize)

    # set plot and legend depending on input type
    if type(profs) == tuple:
        for n in np.arange(len(profs)):
            plt.plot(r, profs[n])

        if not legentries:
            for n in np.arange(len(profs)):
                legentries.append(f"Profile {n+1}")
    else:
        plt.plot(r, profs)
        if not legentries:
            legentries.append("Profile")

    showTicks()
    plt.xlim(r[0], r[-1])

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    if not no_title:
        plt.title(title)
    plt.legend(legentries)

    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)


def interpolationPlot(r: np.ndarray, org: np.ndarray, interpolated: np.ndarray, blocking: bool=True) -> None:
    """
    compare original and interpolated profile curve

    :param r: r vector (1D array)
    :param org: original profile (1D array)
    :param interpolated: interpolated profile (1D array)
    :param blocking: if pyplot pauses further program execution (bool, defaults to true)
    """

    plt.figure(figsize=fsize)
    plt.plot(r, org, r, interpolated)

    plt.xlim(r[0], r[-1])
    showTicks()
    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    if not no_title:
        plt.title("Interpolated Height Profile")
    plt.legend(["Original", "Filtered"])

    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)


def FilteredProfilePlot(r: np.ndarray, org: np.ndarray, filtered: np.ndarray, diff2: np.ndarray, 
                        diff2_filtered: np.ndarray, F: dict, blocking: bool=True) -> None:
    """
    show filtered profile compared to orginal data in one plot,
    abs of second derivative filtered and unfiltered and threshold in second plot

    :param r: r vector (1D array)
    :param org: original profile (1D array)
    :param filtered: filtered profile (1D array)
    :param diff2: unfiltered abs of second derivative (1D array)
    :param diff2_filtered: filtered abs of second derivative (1D array)
    :param F: filter property dictionary, see filterProfile() for more details
    :param blocking: if pyplot pauses further program execution (bool, defaults to true)
    """

    # original and filter plot
    plt.figure(figsize=fsize)
    plt.subplot(211)
    plt.plot(r, org, r, filtered)
    plt.xlim(r[0], r[-1])
    showTicks()

    plt.xlabel("r in µm")
    plt.ylabel("h in µm")
    if not no_title:
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
    if not no_title:
        plt.title("Absolute value of second derivative")
    plt.legend(["prefiltered", "pre- and postfiltered", "threshold"])

    plt.tight_layout()
    plt.show(block=blocking)
    plt.pause(0.1)
