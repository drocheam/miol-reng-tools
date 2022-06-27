#!/usr/bin/env python3

from lib import *
import numpy as np
from pathlib import Path

# Author: Damian Mendroch,
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Generate a mathematical representation of a lens profile using a confocal lens measurement.
Processing includes: Centering, alignment, interpolation, filtering and parameter regression.
The output of this script is a numpy archive (.npz) that holds the radius vector, 
parameters for the refractive base part as well as filtered data of the diffractive part (both 1 dimensional)

User Interaction:
1. specify path of sms and smi file
2. Lens plot
3. Specify lens center and lens diameter using the setLensProperties() Interface
4. Plots of conic section, polynomials and diffractive part
5. Specify if polynomials and diffractive parts will be excluded
6. if diffractive part is included:
a) set interpolation regions for diffractive profile
b) set filtering parameters for diffractive profile (setFiltering() Interface)
7. set saving path for output .npz archive
"""


def GenerateProfiles():
    """
    Processing of a lens measurement to a adjusted, filtered and fitted profile of the lens. The processing settings,
    fitting properties and the processed data are saved in a numpy archive.
    """

    #################################### Data Import and Stitching #####################################################

    path_smt = inputFilePath("Path to sms file: ", ftype=".sms")
    path_smi = inputFilePath("Path to smi file: ", ftype=".smi")
    path = path_smt  # also the path used for saving

    x0, y0, Images = importImages(path_smt)
    SP = getStitchingPreferences(path_smi)

    sr = round(25/(x0[1] -x0[0]))  # search region is 25um in every direction
    arrx, arry = getShiftVectors(Images, ovlp=SP['ovlp'], sr=sr, mode="fft", transpose=SP['upwards'], debug=False)
    x, y, h_data = Stitch(x0, y0, Images, SP, arrx.astype(int), arry.astype(int))

    ################################### Geometric Manipulations #########0##############################################

    # Interpolation of missing data
    h_data_i = interpolateNan(h_data)

    # Lens Plot
    LensPlot(x, y, h_data_i, blocking=False)

    # User specifies lens properties
    LP = setLensProperties(x, y, h_data_i)

    # Cut Lens
    h_data_c = cutLens(x, y, h_data_i, LP['r2'], LP['xm'], LP['ym'])

    # remove outliers
    h_data_c = removeOutliers(h_data_c, 20, 1)

    # compensate lens tilt
    tc = getTiltCorrection(x, y, h_data_c, LP, cut=0.8)
    print(f"\nCorrection tilt x direction: {tc[0]*1000:.3f}mrad ({tc[0]/np.pi*180:.4f}deg)")
    print(f"Correction tilt y direction: {tc[1]*1000:.3f}mrad ({tc[1]/np.pi*180:.4f}deg)")
    r, prof1, prof2 = getProfiles(x, y, h_data_c, LP, tc)
    
    profs = np.nanmean(np.concatenate((prof1, prof2)), axis=0)
    CR = ConicSectionRegression(r, profs, cp=0.75)
    PR = SymPolyRegression(r, profs - ConicSection(r, *CR), order=10)

    # remove offset
    prof1 -= PR[-1]
    prof2 -= PR[-1]
    profs -= PR[-1]
    PR[-1] = 0

    # calculate overall height
    h = profs[0] - profs[-1]

    # print stats
    ProfileInformation(CR, PR, r, h)

    ProfilePlot(r, (ConicSection(r, *CR[:2], k=0), ConicSection(r, *CR), ConicSection(r, *CR) + Poly(r, PR)),
                legentries=["Sphere", "Conic Section", "Conic Section + Polynomial"],
                title="Lens Base Profile", blocking=False)

    ProfilePlot(r, profs- ConicSection(r, *CR) - Poly(r, PR), legentries=["Mean Diffraction Profile"],
                blocking=False, title="Mean Diffraction Profile")

    ProfilePlot(r, Poly(r, PR), legentries=["Polynomial"], blocking=False, title="Polynomial")
    ComparisonPlot(r, prof1, prof2, blocking=False, title="Diffraction Profile")
    ComparisonPlot(r, prof1 - ConicSection(r, *CR)-Poly(r, PR), prof2 - ConicSection(r, *CR) - Poly(r, PR),
                   blocking=True, title="Lens Profile")

    # decide if to use polynomials and profile
    use_poly, use_diff = setProfileUsage()

    if not use_poly:
        PR = []

    if not use_diff:
        filtered = 0
        diff = 0
        F, I = [], []
    else:
        # Profile Interpolation
        diff = profs - ConicSection(r, *CR) - Poly(r, PR)
        prof_f, I = setInterpolation(r, diff)

        # Filtering
        filtered, F = setFiltering(r, prof_f)

        ProfilePlot(r, (diff, filtered), legentries=["Measured Data", "Fitted Data"], 
                    title="Diffraction Profile", blocking=True)
    
    ############################# Save Data ############################################################################
    ppath = Path(path)
    filename = ppath.stem + "_Profile"
    savdict = dict(r=r, h=h, diff=filtered, diff_org=diff, LP=LP, tilt=tc, CR=CR, PR=PR, I=I, F=F)
    saveData(path, filename, savdict)


# execute function when called as main file
if __name__ == "__main__":
    GenerateProfiles()

