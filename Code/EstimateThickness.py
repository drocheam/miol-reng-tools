from lib import *

"""
Estimate the lens edge thickness using a measurement and user inputs.

User Interaction:
1. Specify .nms measurement file path
2. Do until satisfactory results are achieved:
a) user inputs edge lines (more details shown on execution) so they overlay with the edges
b) edge profile and thickness plots
3. Thickness is displayed
"""


def ThicknessEstimation(x=None, y=None, h_data=None):
    """
    estimate the lens thickness at the edge using the distance of the edge lines using user input.
    if x, y or h_data are not specified, a file is imported at runtime

    :param x: x coordinate vector (1D array)
    :param y: y coordinate vector (1D array)
    :param h_data: z values  (2D array)
    """

    # only load measurement if not specified as parameter
    if x is None or y is None or h_data is None:
        path = inputFilePath("Path to nms file: ", ftype=".nms")
        x, y, h_data = importMeasurement(path)

    d = setThicknessBorders(x, y, h_data)
    print("\nThickness: ", "{:.1f}".format(d), "µm")
    print("(assuming the edge diameter line is parallel to the lens axis",
          "and there is no slope towards the lens centre)")


# execute function when called as main file
if __name__ == "__main__":
    ThicknessEstimation()
