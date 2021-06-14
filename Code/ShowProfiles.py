from lib import *

# Author: Damian Mendroch,
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Visualizes processed diffractive MIOL data from a .npz archive.
.npz archive needs to include all data generated from function GenerateProfiles()

Plot of the conic section, polynomials and diffractive profiles are shown,
as well as profile information.

User Interaction:
1. Enter Path to .npz archive
2. Profile plots
"""

def ShowProfiles(path=None):
    """
    loads the generated profile from GenerateProfiles() and shows several properties and plots.
    If path is not given, the user specifies a path at runtime

    :param path: (optional): path to numpy archive
    """

    if path is None:
        path = inputFilePath("Path to npz file: ", ftype=".npz")
    S1 = loadData(path)

    ProfileInformation(S1['CR'], S1['PR'], S1['r'])

    Conic = ConicSection(S1['r'], *S1['CR'])
    ConicCircle = ConicSectionCircle(S1['r'], *S1['CR'])
    Polynomial = Poly(S1['r'], S1['PR'])

    legend1 = ["Profile 1", "Profile 2", "Conic Section + Polynomial"]
    legend2 = ["Conic Section", "Curvature Circle", "Conic Section + Polynomial"]
    legend3 = ["Diffractive Profile 1", "Diffractive Profile 2"]

    hasprofile = S1['diff1'].size > 1 # if diffraction profile needs to be shown

    ProfilePlot(S1['r'], (Conic + S1['diff1'] + Polynomial, Conic + S1['diff2'] + Polynomial, Conic+Polynomial),
                legentries=legend1, blocking=False)
    ProfilePlot(S1['r'], (Conic, ConicCircle, Conic+Polynomial), legentries=legend2, blocking=not hasprofile)

    if hasprofile:
        ProfilePlot(S1['r'], (S1['diff1'], S1['diff2']), legentries=legend3, blocking=True)


# execute function when called as main file
if __name__ == "__main__":
    ShowProfiles()
