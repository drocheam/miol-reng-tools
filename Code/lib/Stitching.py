import copy
import scipy.ndimage
import numpy as np
from matplotlib import pyplot as plt
from lib.Interpolation import interpolateNan

# Author: Damian Mendroch
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Methods for alternative stitching compared to that of nanofocus
shift detection needed because stepper motor used for displacement between images is open loop
      and has no feedback on precise position -> shift detection needed
shift detection for flat surfaces with no details is hard or even impossible,
these methods here are adapted to our purposes and we now have to possibility
to chose the best one from a selection of methods

Steps:
1. load your images from a sms/smt file (using getImages() in Import.py)
   and the stitching preferences from smi file (using getStitchingPreferences() in Import.py)
2. use getShiftVectors() and desired method to determine image shifts
3. use Stitch() and the determined shift vectors to stitch your image

possible methods: (0) stitching from microscope (no restitching needed)  (1) none, assume perfect motor position
(2) shift detection using fft shift detection 

stitching not needed for method (0). since the software already did that for us
"""


def Stitch(x_in: np.ndarray, y_in: np.ndarray, Images_in: list[np.ndarray], SP: dict=dict(),
           xShift: np.ndarray=[], yShift: np.ndarray=[]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    stitch multiple images with specified shift vectors.
    The transition in the overlapping region is smoothed out
    (average of both images, average ratio dependent on position in voerlap region)

    :param x_in: x coordinate vector for one Image (1D array)
    :param y_in: y coordinate vector for one Image (1D array)
    :param Images_in: Images (list of 2D array)
    :param SP: stitching preferences from getStitchingPreferences() (dictionary)
    :param xShift: additional shift in x direction (1D array)
    :param yShift: additional shift in y direction (1D array)
    :return:
    """

    # work on copies
    Images = copy.deepcopy(Images_in)
    x, y = x_in.copy(), y_in.copy()

    # use loaded settings if available
    if len(SP) > 0:

        if SP['shiftY'] != 0:
            raise Exception('shiftY not supported yet')

        if SP['type'] == '2D':
            raise Exception('2D stitching not supported yet')

        ovlp = SP['ovlp'] - (SP['shiftX'] - 432)  # overlap

        # transpose images if they were taken upwards
        if SP['upwards']:
            for n in np.arange(len(Images)):
                Images[n] = Images[n].T
            x, y = y, x
            nImages = SP['ny']

        else:
            nImages = SP['nx']

    # default settings
    else:
        ovlp = 80
        nImages = len(Images)

    # set shifts to zero when Shifts are missing
    if len(xShift) == 0:
        xShift = np.zeros(nImages-1, dtype=int)

    if len(yShift) == 0:
        yShift = np.zeros(nImages-1, dtype=int)

    h_data = Images[0]  # first image
    x0 = x_in.copy()  # x vector for first image

    hmin, hmax = 0, h_data.shape[0]  # coordinates of y start and y end of the stitched image
    imin, imax = 0, h_data.shape[0]  # coordinates of y start and y end of the new image

    for n in np.arange(nImages-1):

        Image = Images[n+1]  # current image
        ol = ovlp - xShift[n]  # overlap region size in x direction

        # shift new image coordinates relatively
        imin -= yShift[n]
        imax -= yShift[n]

        if imin > hmin:  # y start of current Image is above the y start of the stitched image
            h_data = h_data[imin-hmin:, :]  # slice stitched image to overlapping y region
            y = y[imin-hmin:]  # slice y vector to overlapping region
            hmin = imin  # update hmin

        if imax < hmax:  # y end of current Image is below the y end of the stitched image
            h_data = h_data[:imax-hmax, :]  # slice stitched image to overlapping y region
            y = y[:imax-hmax]  # slice y vector to overlapping region
            hmax = imax  # update hmax

        Image = Image[hmin-imin:hmax-imin, :] # slice new image to overlapping y region

        left = h_data[:, -ol:]  # overlap region of left image
        right = Image[:, :ol]  # overlap region of right image

        # use z offset if available
        offset = np.nanmedian(left - right)  # usurf stitching also uses median for offset detection
        if ~np.isnan(offset):  # if finite use offset
            Image += offset

        # smooth data in overlap regions (similar or maybe even equal to what usurf software is doing)
        scale = np.arange(ol) / ol
        Image[:, :ol] = left * (1 - scale) + right * scale  # gliding average
        Image[:, :ol][np.isnan(left)] = right[np.isnan(left)]  # copy right data when left is missing
        Image[:, :ol][np.isnan(right)] = left[np.isnan(right)]  # copy left data when right is missing

        # append data
        h_data = np.concatenate((h_data[:, :-ol], Image), axis=1)
        x = np.concatenate((x[:-ol], x0 + x[-ol]))

    return x, y, h_data


def getShiftVectors(Images_in: list[np.ndarray], mode: str='none', ovlp: int=80, sr: int=40,
                    transpose: bool=False, debug: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    use selected method to generate shift vectors between images

    :param Images_in: Image list (list of 2D arrays)
    :param mode: method for detection, "none", "fft"
    :param ovlp: size of overlap region (int)
    :param sr: search region for shift (int)
    :param transpose: if data needs to be transposed for processing (bool)
    :param debug: if debug mode is set for methods (bool)
    :return: suggested shift x direction (np.ndarray), suggested shift y direction (np.ndarray)
    """

    # transpose if images lie in y direction
    if transpose:
        Images = copy.deepcopy(Images_in)  # deepcopy needed so input parameter reference is not changed
        for n in np.arange(len(Images)):
            Images[n] = Images[n].T
    else:
        Images = Images_in

    # default shift vectors
    arrx = np.zeros(len(Images) - 1)
    arry = np.zeros(len(Images) - 1)

    # change method depending on method string
    if mode == 'fft':
        for n in np.arange(len(Images) - 1):
            arrx[n],  arry[n] = getShiftFFT(Images[n][:, -ovlp:], Images[n+1][:, :ovlp], sr, debug=debug)[:2]

    elif mode != 'none':
        raise Exception("Invalid mode")

    # on mode == "none" nothing needs to be done, since default vectors values are already set

    return arrx, arry


def getShiftFFT(Image1: np.ndarray, Image2: np.ndarray, sr: int=40, debug: bool=False) -> tuple[int, int, bool]:
    """
    find shift vector using fft method, back transformed ratio of the fft of two shifted images yields a dirac pulse
    at the position of the shift.
    The shift is only valid, if it lies within the search region -sr to +sr in both directions,
    if the images have sufficent valid data and when the dirac pulse is distintive enough (6 sigma above mean)

    :param Image1: left Image (2D array)
    :param Image2: right Image (2D array)
    :param sr: search region (int)
    :param debug: if stitching information and plots are shown in the process (bool, default is false)
    :return: shift x direction (float), shift y direction (float), algorithm result valid (bool)
    """

    # return zero shift if insufficient finite data
    if np.count_nonzero(np.isfinite(Image1) & np.isfinite(Image2)) < 0.8*Image1.size:
        if debug:
            print("Skipping Image, insufficient finite data")
        return 0, 0, False

    # replace nan values
    ImA = np.nan_to_num(interpolateNan(Image1), nan=np.nanmean(Image1))
    ImB = np.nan_to_num(interpolateNan(Image2), nan=np.nanmean(Image2))

    # quadrant shift, from (0, 0) in the upper left corner to the image center
    A = np.fft.fftshift(ImA)
    B = np.fft.fftshift(ImB)

    # 2D Fourier Transform with orthogonal norm
    A_F = np.fft.fft2(A, norm='ortho')
    B_F = np.fft.fft2(B, norm='ortho')

    # image h1(x,y) and shifted image h2(x,y) ≈ h1(x-sx,y-sy) in spatial domain
    # become H1(fx,fy) and H2(fx, fy) ≈ H1(fx,fy)*e^(-i*2*pi*(fx*sx + fy*sy)) in the fourier domain
    # due to the shift theorem, isolate exponential function using the ratio Q = H2/H1
    Q = np.zeros(A_F.shape, dtype=complex)
    Q[A_F != 0] = B_F[A_F != 0] / A_F[A_F != 0]

    # e^(-i*2*pi*(fx*sx + fy*sy)) reverse transformed is dirac(x-sx, y-sy), so we need to find the maximum position
    V = np.fft.ifft2(Q, norm='ortho')  # reverse transformation
    V = np.fft.ifftshift(V)  # reverse quadrant shift
    Vf = scipy.ndimage.gaussian_filter(np.abs(V), 1)  # filter result

    # cut to search region sr
    iy, ix = ImA.shape
    Vfc = Vf[iy//2-sr:iy//2+sr, ix//2-sr:ix//2+sr]

    # find peak
    k_y, k_x = np.where(Vfc == np.max(Vfc))

    # only valid if maximum found and if distinctive enough (6*sigma above mean)
    valid = k_x.shape[0] == 1 and np.max(Vfc) > np.mean(Vf) + 6*np.std(Vf)

    if valid:
        v_y, v_x = int(sr - k_y), int(k_x - sr)  # calculate shift values from coordinates
    else:
        v_y, v_x = 0, 0 # default shift

    # note that v_x and v_y are the shift between Image1 and Image2, to correct the shift we must use -v_x, -v_y
    # that's why the negative values are returned

    if debug:
        varsh, ImAs, ImBs = ImVar(ImA, ImB, -v_x, -v_y) # variance and images at best shift vector
        var0 = np.nanvar(Image1-Image2) # variance without shift applied

        print(f"Valid result: {valid}")
        print(f"Detected shift: v_x = {v_x}, v_y = {v_y}")
        print(f"variance before shift: {var0}, variance after shift: {varsh}")

        FFTDebugPlot(ImA, ImB, Vfc, sr, ImAs, ImBs, v_x, v_y)

    return -v_x, -v_y, valid


def ImVar(Image1_in: np.ndarray, Image2_in: np.ndarray, v_x: float, v_y: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    calculate variance at the overlap of two shifted images
    :param Image1_in: left Image (2D array)
    :param Image2_in: right Image (2D array)
    :param v_x: shift x direction (float)
    :param v_y: shift y direction (float)
    :return: variance (float), as well as overlap regions of shifted images (both 2D array)
    """

    # directions in pythonic coordinate system, meaning:
    # v_x > 0: Image2 right of Image 1; v_y > 0: Image2 below of Image1

    # choose x slice
    if v_x > 0:     xr1, xr2 = slice(v_x, None), slice(-v_x)
    elif v_x < 0:   xr1, xr2 = slice(v_x), slice(-v_x, None)
    else:           xr1, xr2 = slice(None), slice(None)

    # chose y slice
    if v_y > 0:     yr1, yr2 = slice(-v_y), slice(v_y, None)
    elif v_y < 0:   yr1, yr2 = slice(-v_y, None), slice(v_y)
    else:           yr1, yr2 = slice(None), slice(None)

    # slice images
    if v_x or v_y:  Image1, Image2 = Image1_in[yr1, xr1], Image2_in[yr2, xr2]
    else:           Image1, Image2 = Image1_in.copy(), Image2_in.copy()  # no slicing, make copy of inputs

    diff = Image1 - Image2
    valid = np.isfinite(diff)
    valid_count = np.count_nonzero(valid)

    if valid_count > 20:
        variance = np.var(diff[valid]) * Image1_in.size / valid_count  # scale with data quantity
    else:
        variance = np.nan

    return variance, Image1, Image2


def FFTDebugPlot(ImA: np.ndarray, ImB: np.ndarray, Vf: np.ndarray, sr: int, ImAs, ImBs, v_x: int, v_y: int) -> None:
    """
    plotting function that is called from getShiftFFT()

    :param ImA: left image data (2D array)
    :param ImB: right image data (2D array)
    :param Vf: variance image (2D array)
    :param sr: search region (int)
    :param ImAs: left image shifted data (2D array)
    :param ImBs: right Image shifted data (2D array)
    :param v_x: best shift value x (int)
    :param v_y: best shift value y (int)
    """

    # for plotting the mathematical coordinate system instead of the pythonic is used,
    # meaning v_y and images need to be flipped (e.g. with the [::-1, :] operator)
    # the axis extent also needs to be set

    # Image extents
    extA = [0, ImA.shape[1], 0, ImA.shape[0]]
    extB = [0, ImAs.shape[1], 0, ImAs.shape[0]]

    plt.figure(figsize=(8, 8))

    plt.subplot(151)
    plt.imshow(ImA[::-1, :], cmap='magma', extent=extA)
    plt.title('Image1')

    plt.subplot(152)
    plt.imshow(ImB[::-1, :], cmap='magma', extent=extA)
    plt.title('Image2')

    plt.subplot(153)
    plt.imshow(np.abs(Vf)[::-1, :], cmap='magma', extent=[-sr, sr, -sr, sr])
    plt.scatter(v_x, -v_y, color='white', marker='x', s=50)
    plt.title('V')

    plt.subplot(154)
    plt.imshow(ImAs[::-1,: ], cmap='magma', extent=extB)
    plt.title('Image1 shifted')

    plt.subplot(155)
    plt.imshow(ImBs[::-1, :], cmap='magma', extent=extB)
    plt.title('Image2 shifted')
    plt.show()
