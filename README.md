# miol-reng-tools
## Tools for Reverse Engineering Multifocal Intraocular Lenses

This is the repository of the project *Reverse Engineering of Multifocal Intraocular Lenses Using a Confocal Microscope* that was conducted as research seminar at the TH Köln in WS 2020/21.

Details can be found in the [paper](https://github.com/drocheam/miol-reng-tools/blob/main/Paper.pdf) included in the repository.

### Screenshots
 <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot2.png" height="220">  <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot1.png" height="220">  

### Structure

Element | Function
------------ | -------------
`Measurements\` | Measurement data and processing parameters
`Code\lib\` | Implemented library package
`Code\GenerateProfiles.py` | Generate surface profiles from a measurement
`Code\EstimateThickness.py` | Measure lens edge thickness
`Code\CompareStitching.py` | Compare different stitching methods on a measurement
`Code\ShowProfiles.py` |  Show generated profiles

The project includes functions from [musurf-reader](https://github.com/drocheam/musurf-reader).
<br></br>

### Required packages:
`copy, matplotlib, numpy, PyQt5, scipy, struct, os`

Tested with Python 3.8, PyCharm 2020.3.2 and Spyder 4.2.1
