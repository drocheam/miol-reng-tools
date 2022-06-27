from lib.Plotting import *
from lib.Filtering import filterProfile
from lib.Interpolation import interpolateProfile
import numpy as np
import os.path
from typing import Callable
from sys import exit

# Author: Damian Mendroch
# Project repository: https://github.com/drocheam/miol-reng-tools

"""
Interface functions.
Inputs are checked for validity and actions are performed on the input until the user is satisfied with the results.
type "end" to exit interfaces.

"""


def inputLoop(info: str, A: list | dict, func: Callable, func2: Callable=(lambda a: a)) -> tuple[str, dict]:
    """
    asks for input until a valid input is given (specified by lambdas) or input equals "end"

    :param info:  message for each input iteration (string)
    :param A: processed input (type is specified by func and func2 lambdas)
    :param func: lambda for processing the input to A
    :param func2: (optional): second lambda for processing the input to A
    :return: raw input (string) and processed input A
    """

    valid = False
    inp = ""
    while valid == False and inp != "end":
        inp = input(info)
        if inp == "exit":
            exit()
        try:

            if inp != "end":
                A = func2(func(inp))
                valid = True
        except:
            print("Invalid Input")

    return inp, A


def inputFilePath(message: str="Enter path to file: ", ftype: str="all") -> str:
    """
    prompts for a file path and checks if file exists and if the filetype is correct

    :param message: message to be shown (string)
    :param ftype: tolerated filetype (string, e.g. ".txt", default to "all")
    :return: valid file path (string)
    """

    while 1:
        path = input(message)
        path = path.replace("\"", "")
        ext = os.path.splitext(path)[1]

        if ftype != 'all' and ext != ftype:
            print(f"Filetype needs to be {ftype}")

        elif os.path.isfile(path):
            return path

        else:
            print("File not Found")


def saveData(path: str, name: str, tosave: dict) -> str:
    """
    saves dictionary as npz archive, checks if file exists and asks user if he wants it to be replaced

    :param path: folder or filepath used for saving (string)
    :param name: file name for saving without extension (string)
    :param tosave: dictionary to save
    :return: full path of saved archive (string)
    """

    folder = os.path.split(path)[0]
    outpath = os.path.join(folder, name + ".npz")

    while os.path.isfile(outpath):
        choice = input("\nFile exists. Overwrite? (y/n) ")
        if choice == 'Y' or choice == 'y':
            break
        elif choice == 'N' or choice == 'n':
            print("Saving Aborted")
            return ""
        print("Invalid Input")

    np.savez(outpath, **tosave)
    print(f"\nSaved Data as {outpath}")
    return outpath


def loadData(path: str) -> np.ndarray:
    """
    loads and returns data from .np or .npz archive (string)

    :param path: filepath (string)
    :return: data from .np or .npz archive
    """
    return np.load(path)


def setLensProperties(x: np.ndarray, y: np.ndarray, h_data: np.ndarray) -> dict:
    """
    Interface for setting the lens properties, the user can adjust the properties mutiple times to his liking
    for each iteration the DiffLensPlot is shown

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data: z values (2D array)
    :return: lens property dictionary, containing the specified xm, ym, r1, r2 values
    """

    print("\nSet Lens Properties. xm, ym Coordinates of Lens Axis, r1 Radius of Aid Circle, r2 Lens Radius."
          , "Type \"end\" to exit")

    S = []
    inp = ""

    while inp != "end":
        try:
            DiffLensPlot(x, y, h_data, S)
        except Exception as e:
            print("Exception:", str(e))

        func = lambda b: list(map(int, b.strip().split()))[:4]
        func2 = lambda a:  dict(xm=a[0], ym=a[1], r1=a[2], r2=a[3])
        inp, S = inputLoop("S = xm ym r1 r2: ", S, func, func2)

    return S


def setInterpolation(r: np.ndarray, prof: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Interface for profile interpolation, the user can set ranges for linear interpolation.
    This whole process is optional. For each iteration the orginal and interpolated data is shown

    :param r: r vector (1D array)
    :param prof: profile vector (1D array)
    :return: the interpolated profile (1D array), the interpolation range start and end points (1D array)
    """

    print("\nEnter Interpolation Sections as Data Pairs (optional). Type \"end\" to exit")

    I = []
    prof_f = []
    inp = ""

    while inp != "end":
        try:
            prof_f = interpolateProfile(r, prof, I, 'linear')
            interpolationPlot(r, prof, prof_f)
        except Exception as e:
            print("Exception:", str(e))

        func = lambda b: list(map(float, b.strip().split()))[:]
        inp, I = inputLoop("I = x11 x12 x21 x22 ...]: ", I, func)

    return prof_f, I


def setProfileUsage() -> tuple[bool, bool]:
    """
    Interface for setting if polynomial and differential data are used in further processing

    :return: two boolean values (poly usage and diff usage)
    """
    B = [True, True]

    print("\nUse Polynomial Curve? (b1 = 0/1) Use Difference Profile? (b2 = 0/1)")

    func = lambda b: list(map(int, b.strip().split()))[:2]
    _, B = inputLoop("B = b1 b2: ", B, func)

    return bool(B[0]), bool(B[1])


def setFiltering(r: np.ndarray, prof: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Interface for profile filtering using the second derivative and filter sections.
    For each iteration the orginal and filtered data is shown, as well as the second derivative.

    :param r: r vector (1D array)
    :param prof: profile vector (1D array)
    :return: filtered data (1D array), filter property dictionary (see code)
    """

    F = dict(fc1=5, fc2=7, tr=0.005, n=4, xb=[])
    filtered = []
    inp = ""

    print("\nEnter Filtering Parameters  F = fc1 fc2 tr n (xb1, xb2, ...). Type \"end\" to exit.")
    print(f"(Default: F = {F}")
    print("fc1: gaussian filter parameter for filtering before differentiation,")
    print("fc2: gaussian filter parameter for filtering after differentiation,")
    print("tr: threshold for section division,", "n: polynomial spline order,")
    print("xb1, xb2 (optional): points, at which additional sections are created")

    while inp != "end":
        try:
            filtered, d2, d2f = filterProfile(r, prof, F)
            FilteredProfilePlot(r, prof, filtered, d2, d2f, F)
        except Exception as e:
            print("Exception:", str(e))

        func = lambda b: list(map(float, b.strip().split()))[:]
        func2 = lambda a:  dict(fc1=a[0], fc2=a[1], tr=a[2], n=a[3], xb=a[4:])
        inp, F = inputLoop("F = fc1 fc2 tr n (xb1, xb2, ...): ", F, func, func2)

    return filtered, F
