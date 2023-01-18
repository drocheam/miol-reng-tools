# miol-reng-tools
## Tools for Reverse Engineering Multifocal Intraocular Lenses

Repository of the project *Reverse Engineering of Multifocal Intraocular Lenses Using a Confocal Microscope* that was conducted as research seminar at the TH Köln in WS 2020/21.

Measurement and processing results are featured in:<br>
*Mendroch, D., Altmeyer, S. & Oberheide, U. Characterization of diffractive bifocal intraocular lenses. Sci Rep 13, 908 (2023).* https://doi.org/10.1038/s41598-023-27521-7


### Screenshots
 <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot2.png" height="220">  <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot1.png" height="220">  

### Structure

Element | Function
------------ | -------------
`Measurements\` | Sample measurement data
`Code\lib\` | Implemented library package
`Code\GenerateProfiles.py` | Generate surface profiles from a measurement
`Code\ShowProfiles.py` |  Show processed profiles

The project includes functions from [musurf-reader](https://github.com/drocheam/musurf-reader).
<br></br>

### Required packages:
`copy, matplotlib, numpy, PyQt5, scipy, struct, os`

Tested with Python 3.10

### Usage
Measurement files from a nanofocus µsurf custom confocal microscope are required (smt and smi format). 

A sample measurement can be found under `Measurements\`.
