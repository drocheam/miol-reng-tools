from lib import *
import numpy as np

"""
Generate a mathematical representation of a lens profile using a confocal lens measurement.
Processing includes: Centering, alignment, interpolation, filtering and parameter regression.
The output of this script is a numpy archive (.npz) that holds the radius vector, 
parameters for the refractive base part as well as filtered data of the diffractive part (both 1 dimensional)

User Interaction:
1. Chose stiching method
2. Depending on stitching method specify path to nms or sms and smi file
3. Lens plot
4. Specify lens center and lens diameter using the setLensProperties() Interface
5. Plots of conic section, polynomials and diffractive part
6. Specify if polynomials and diffractive parts will be excluded
7. if diffractive part included: do for both lens surface sides:
a) set interpolation regions for diffractive profile
b) set filtering parameters for diffractive profile (setFiltering() Interface)
8. set saving path for output .npz archive
"""


def GenerateProfiles(x=None, y=None, h_data=None):
    """
    Processing of a lens measurement to a adjusted, filtered and fitted profile of the lens. The processing settings,
    fitting properties and the processed data is saved in a numpy archive.
    If this function is called without parameters, a measurement file is imported at runtime

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data: z values  (2D array)
    """

    #################################### Data Import and Stitching #####################################################

    # no data given
    if x is None or y is None or h_data is None:

        SM = setStitching()
        if SM == "usurf":
            path = inputFilePath("Path to nms file: ", ftype=".nms")
            x, y, h_data = importMeasurement(path)
        else:
            path_smt = inputFilePath("Path to sms file: ", ftype=".sms")
            path_smi = inputFilePath("Path to smi file: ", ftype=".smi")
            path = path_smt # also path used for saving

            x0, y0, Images = importImages(path_smt)
            SP = getStitchingPreferences(path_smi)

            sr = round(25/(x0[1] -x0[0])) # search region is 25um in every direction
            arrx, arry = getShiftVectors(Images, ovlp=SP['ovlp'], sr=sr, mode=SM, transpose=SP['upwards'], debug=False)
            x, y, h_data = Stitch(x0, y0, Images, SP, arrx.astype(int), arry.astype(int))
    # data given
    else:
        path = "./"
        SM = "not specified"

    ################################### Geometric Manipulations #########0###############################################

    # Interpolation of missing data
    h_data_i = interpolateNan(h_data)

    # Lens Plot
    LensPlot(x, y, h_data_i)

    # User specifies lens properties
    LP = setLensProperties(x, y, h_data_i)

    # Cut Lens
    h_data_c = cutLens(x, y, h_data_i, LP['r2'], LP['xm'], LP['ym'])

    # compensate lens tilt
    radX, radY = tiltRegression(x, y, h_data_c, LP, cp=-1)
    print("\nTilt x direction:", "{:.3f}".format(radX*1000), "mrad (", "{:.4f}".format(radX/np.pi*180), "deg)")
    print("Tilt y direction:", "{:.3f}".format(radY*1000), "mrad (", "{:.4f}".format(radY/np.pi*180), "deg)")

    # Generate 1D Profiles from 2D data
    r, prof1, prof2 = getProfiles(x, y, h_data_c, LP, [-radX, -radY])

    profs = np.nanmean(np.concatenate((prof1, prof2)), axis=0)
    CR = ConicSectionRegression(r, profs, cp=0.75)
    PR = SymPolyRegression(r, profs- ConicSection(r, *CR), order=8)

    ProfileInformation(CR, PR, r)

    # Left and Right Profile Comparisons
    ProfilePlot(r, (ConicSection(r, *CR), ConicSection(r, *CR)+Poly(r, PR)),
                legentries=["Conic Section", "Conic Section + Polynomial"], title="Conic Section", blocking=False)
    ProfilePlot(r, Poly(r, PR), legentries=["Polynomial"], blocking=False, title="Polynomial")
    ComparisonPlot(r, prof1 - ConicSection(r, *CR)-Poly(r, PR), prof2 - ConicSection(r, *CR)-Poly(r, PR),
                   blocking=True, title="Diffraction Profile")

    # decide if to use polynomials and profile
    use_poly, use_diff = setProfileUsage()

    if not use_poly:
        PR = []

    if not use_diff:
        filtered1, filtered2 = 0, 0
        F1, F2, I1, I2 = [], [], [], []
    else:
        ######################### Filtering Left Profile ###############################################################

        # Profile Interpolation
        diff1 = np.mean(prof1, axis=0) - ConicSection(r, *CR) - Poly(r, PR)
        prof_f1, I1 = setInterpolation(r, diff1)

        # Filtering
        filtered1, F1 = setFiltering(r, prof_f1)

        ######################### Filtering Right Profile ##############################################################

        # Profile Interpolation
        diff2 = np.mean(prof2, axis=0) - ConicSection(r, *CR) - Poly(r, PR)
        prof_f2, I2 = setInterpolation(r, diff2)

        # Filtering
        filtered2, F2 = setFiltering(r, prof_f2)

    ############################# Save Data ############################################################################
    savdict = dict(r=r, diff1=filtered1, diff2=filtered2, SM=SM, LP=LP, tilt=[-radX, -radY], CR=CR, PR=PR, I1=I1, I2=I2, F1=F1, F2=F2)
    saveData(path, "Profile", savdict)


# execute function when called as main file
if __name__ == "__main__":
    GenerateProfiles()
