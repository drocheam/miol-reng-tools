from lib import *

# Author: Damian Mendroch,
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Processes a measurement and shows results for every available stitching method.

User Interaction:
1. Specify measurement file paths: nms file, sms file, smi file
2. For every stitching method:
    Specify lens centre and lens diameter using the setLensProperties() Interface (same as in GenerateProfiles())
3. Stitching Plots
"""


def CompareStitching():
    """
    compare stitching methods ("usurf", "none", "fft", "var") on a measurement by plotting the profiles,
    for this the lens properties need to be specified for each result at runtime
    """

    # initialize
    modes = ["usurf", "none", "fft", "var"]
    x, y, h_data, LP = [], [], [], []

    # load data
    path_nms = inputFilePath("Enter path to nms file: ", ftype=".nms") # nms path
    path_smt = inputFilePath("Enter path to sms file: ", ftype=".sms") # sms path
    path_smi = inputFilePath("Enter path to smi file: ", ftype=".smi") # smi path
    SP = getStitchingPreferences(path_smi)

    # load stitched image
    x0, y0, Images = importImages(path_smt)

    # load and stitch all images
    for n in np.arange(len(modes)):
        if modes[n] == "usurf":
            x_i, y_i, h_data_i = importMeasurement(path_nms)
        else:
            sr = round(25/(x0[1] -x0[0])) # search region for stitching shift is ±25µm
            arrx, arry = getShiftVectors(Images, ovlp=SP['ovlp'], sr=sr, mode=modes[n],
                                         transpose=SP['upwards'], debug=False)
            x_i, y_i, h_data_i = Stitch(x0, y0, Images, SP, arrx.astype(int), arry.astype(int))

        x.append(x_i)
        y.append(y_i)
        h_data.append(h_data_i)

    # interpolate and set lens axis
    for n in np.arange(len(modes)):
        h_data[n] = interpolateNan(h_data[n]) # Interpolation of missing data
        LP.append(setLensProperties(x[n], y[n], h_data[n]))  # User specifies lens properties

    # cut, tilt and fit profiles
    for n in np.arange(len(modes)):

        # Cut Lens
        h_data_c =   cutLens(x[n], y[n], h_data[n], LP[n]['r2'], LP[n]['xm'], LP[n]['ym'])

        # compensate lens tilt
        radX, radY = tiltRegression(x[n], y[n], h_data_c, LP[n], cp=-1)

        # Generate 1D Profiles from 2D data
        r, prof1, prof2 = getProfiles(x[n], y[n], h_data_c, LP[n], [-radX, -radY])

        # Calculate nearest asphere
        profs = np.nanmean(np.concatenate((prof1, prof2)), axis=0)
        AR = ConicSectionRegression(r, profs, cp=0.75)
        PR = SymPolyRegression(r, profs- ConicSection(r, *AR))

        # Left and Right Profile Comparisons
        ComparisonPlot(r, prof1 - ConicSection(r, *AR)-Poly(r, PR), prof2 - ConicSection(r, *AR)-Poly(r, PR),
                       blocking=(n==len(modes)-1), title=modes[n])


# execute function when called as main file
if __name__ == "__main__":
    CompareStitching()
