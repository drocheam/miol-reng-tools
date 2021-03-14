# `miol-reng-tools` - Tools for Reverse Engineering Multifocal Intraocular Lenses

This is the repository of the project *Reverse Engineering of Multifocal Intraocular Lenses Using a Conventional Confocal Microscope* that was conducted as research seminar at the TH Köln in WS 2020/21.

Details can be found in the paper included in the repository.

### Structure

Element | Function
------------ | -------------
`Measurements\` | measurement data and processing parameters
`Code\lib\` | implemented library package
`Code\GenerateProfiles.py` | Generate surface profiles from a measurement
`Code\EstimateThickness.py` | Measure lens edge thickness
`Code\CompareStitching.py` | Compare different stitching methods on a measurment
`Code\ShowProfiles.py` |  Show generated profiles

### Required packages:
`copy, matplotlib, numpy, scipy, struct, os`

Tested with Python 3.8 and PyCharm 2020.3.2
