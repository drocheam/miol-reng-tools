#!/usr/bin/env python3

from lib import *

# Author: Damian Mendroch
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Visualizes processed diffractive MIOL data from a .npz archive.
.npz archive needs to include all data generated from function GenerateProfiles()

Plot of the conic section, polynomials and diffractive profiles are shown,
as well as profile information.

"""


def ShowProfiles():
    """
    loads the generated profile from GenerateProfiles() and shows several properties and plots.
    """

    path = inputFilePath("Path to npz file: ", ftype=".npz")
    S1 = loadData(path)

    ProfileInformation(S1['CR'], S1['PR'], S1['r'], S1['h'])

    Conic = ConicSection(S1['r'], *S1['CR'])
    ConicCircle = ConicSectionCircle(S1['r'], *S1['CR'])
    Polynomial = Poly(S1['r'], S1['PR'])

    legend1 = ["Profile", "Conic Section + Polynomial"]
    legend2 = ["Conic Section", "Curvature Circle", "Conic Section + Polynomial"]
    legend3 = ["Mean Profile", "Fitted Profile"]

    hasprofile = S1['diff'].size > 1 # if diffraction profile needs to be shown

    ProfilePlot(S1['r'], (Conic + S1['diff_org'] + Polynomial, Conic+Polynomial),
                legentries=legend1, blocking=False)
    ProfilePlot(S1['r'], (Conic, ConicCircle, Conic+Polynomial), legentries=legend2, blocking=not hasprofile)

    if hasprofile:
        ProfilePlot(S1['r'], (S1['diff_org'], S1['diff']), legentries=legend3, blocking=True)


# execute function when called as main file
if __name__ == "__main__":
    ShowProfiles()

